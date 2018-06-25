// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TensorFlow operations for dynamic batching.

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace {

REGISTER_OP("Batcher")
    .Output("handle: resource")
    .Attr("minimum_batch_size: int")
    .Attr("maximum_batch_size: int")
    .Attr("timeout_ms: int")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
A Batcher which batches up computations into the same batch.
)doc");

REGISTER_OP("BatcherCompute")
    .Input("handle: resource")
    .Input("input_list: Tinput_list")
    .Attr("Tinput_list: list(type) >= 1")
    .Attr("Toutput_list: list(type) >= 1")
    .Output("output_list: Toutput_list")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Puts the input into the computation queue, waits and returns the result.
)doc");

REGISTER_OP("BatcherGetInputs")
    .Input("handle: resource")
    .Attr("Toutput_list: list(type) >= 1")
    .Output("output_list: Toutput_list")
    .Output("computation_id: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_outputs() - 1; ++i) {
        c->set_output(i, c->UnknownShape());
      }
      return c->set_output("computation_id", {c->Scalar()});
    })
    .Doc(R"doc(
Gets a batch of inputs to compute the results of.
)doc");

REGISTER_OP("BatcherSetOutputs")
    .Input("handle: resource")
    .Input("input_list: Tinput_list")
    .Input("computation_id: int64")
    .Attr("Tinput_list: list(type) >= 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Sets the outputs of a batch for the function.
)doc");

REGISTER_OP("BatcherClose")
    .Input("handle: resource")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Closes the batcher and cancels all pending batcher operations.
)doc");

class Batcher : public ResourceBase {
 public:
  using DoneCallback = AsyncOpKernel::DoneCallback;

  Batcher(int32 minimum_batch_size, int32 maximum_batch_size,
          gtl::optional<std::chrono::milliseconds> timeout)
      : ResourceBase(),
        curr_computation_id_(0),
        is_closed_(false),
        minimum_batch_size_(minimum_batch_size),
        maximum_batch_size_(maximum_batch_size),
        timeout_(std::move(timeout)) {}

  string DebugString() override {
    mutex_lock l(mu_);
    return strings::StrCat("Batcher with ", inputs_.size(), " waiting inputs.");
  }

  void Compute(OpKernelContext* context, const OpInputList& input_list,
               DoneCallback callback);

  void GetInputs(OpKernelContext* context, OpOutputList* output_list);

  void SetOutputs(OpKernelContext* context, const OpInputList& input_list,
                  int64 computation_id);

  void Close(OpKernelContext* context);

 private:
  class Input {
   public:
    Input(OpKernelContext* context, const OpInputList& input_list,
          DoneCallback callback)
        : context_(context),
          input_list_(input_list),
          callback_(std::move(callback)) {}

    // Moveable but not copyable.
    Input(Input&& rhs)
        : Input(rhs.context_, rhs.input_list_, std::move(rhs.callback_)) {
      rhs.context_ = nullptr;  // Mark invalid.
    }

    Input& operator=(Input&& rhs) {
      this->context_ = rhs.context_;
      this->input_list_ = rhs.input_list_;
      this->callback_ = std::move(rhs.callback_);
      rhs.context_ = nullptr;  // Mark invalid.
      return *this;
    }

    OpKernelContext* context() const {
      CHECK(is_valid());
      return context_;
    }

    const OpInputList& input_list() const {
      CHECK(is_valid());
      return input_list_;
    }

    bool is_valid() const { return context_ != nullptr; }

    void Done() {
      CHECK(is_valid());

      // After callback is called, context_, input_list_ and callback_ becomes
      // invalid and shouldn't be used.
      context_ = nullptr;
      callback_();
    }

   private:
    // Not owned.
    OpKernelContext* context_;
    OpInputList input_list_;
    DoneCallback callback_;
  };

  void CancelInput(Input* input) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void GetInputsInternal(OpKernelContext* context, OpOutputList* output_list)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void SetOutputsInternal(OpKernelContext* context,
                          const OpInputList& input_list, int64 computation_id)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Cancels all pending Compute ops and marks the batcher closed.
  void CancelAndClose(OpKernelContext* context) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  condition_variable full_batch_or_cancelled_cond_var_;

  // A counter of all batched computations that have been started that is used
  // to create a unique id for each batched computation.
  int64 curr_computation_id_ GUARDED_BY(mu_);

  // Inputs waiting to be computed.
  std::deque<Input> inputs_ GUARDED_BY(mu_);

  // Batches that are currently being computed. Maps computation_id to a batch
  // of inputs.
  gtl::FlatMap<int64, std::vector<Input>> being_computed_ GUARDED_BY(mu_);

  // Whether the Batcher has been closed (happens when there is an error or
  // Close() has been called.)
  bool is_closed_ GUARDED_BY(mu_);
  const int32 minimum_batch_size_;
  const int32 maximum_batch_size_;
  const gtl::optional<std::chrono::milliseconds> timeout_;

  TF_DISALLOW_COPY_AND_ASSIGN(Batcher);
};

void Batcher::Compute(OpKernelContext* context, const OpInputList& input_list,
                      DoneCallback callback) {
  bool should_notify;

  {
    mutex_lock l(mu_);

    OP_REQUIRES_ASYNC(context, !is_closed_,
                      errors::Cancelled("Batcher is closed"), callback);

    // Add the inputs to the list of inputs.
    inputs_.emplace_back(context, input_list, std::move(callback));

    should_notify = inputs_.size() >= minimum_batch_size_;
  }

  if (should_notify) {
    // If a GetInputs operation is blocked, wake it up.
    full_batch_or_cancelled_cond_var_.notify_one();
  }
}

void Batcher::GetInputs(OpKernelContext* context, OpOutputList* output_list) {
  CancellationManager* cm = context->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();

  bool is_cancelled_or_cancelling = !cm->RegisterCallback(
      token, [this]() { full_batch_or_cancelled_cond_var_.notify_all(); });

  mutex_lock l(mu_);
  std::cv_status status = std::cv_status::no_timeout;

  // Wait for data if the input list has fewer samples than `minimum_batch_size`
  // (or non-empty when a timeout has occurred), for cancellation of the
  // operation or for the batcher to be closed.
  while (((status == std::cv_status::timeout && inputs_.empty()) ||
          (status == std::cv_status::no_timeout &&
           inputs_.size() < minimum_batch_size_)) &&
         !is_cancelled_or_cancelling && !is_closed_) {
    // Using a timeout to make sure the operation always completes after a while
    // when there isn't enough samples and for the unlikely case where the
    // operation is being cancelled between checking if it has been cancelled
    // and calling wait_for().
    if (timeout_) {
      status = full_batch_or_cancelled_cond_var_.wait_for(l, *timeout_);
    } else {
      // Timeout is only used to check for cancellation as described in the
      // comment above.
      full_batch_or_cancelled_cond_var_.wait_for(
          l, std::chrono::milliseconds(100));
    }
    is_cancelled_or_cancelling = cm->IsCancelled();
  }

  if (is_closed_) {
    context->SetStatus(errors::Cancelled("Batcher is closed"));
  } else if (is_cancelled_or_cancelling) {
    context->SetStatus(errors::Cancelled("GetInputs operation was cancelled"));
  } else {
    GetInputsInternal(context, output_list);
  }

  if (!context->status().ok()) {
    CancelAndClose(context);
  }
}

void Batcher::GetInputsInternal(OpKernelContext* context,
                                OpOutputList* output_list) {
  int64 batch_size = std::min<int64>(inputs_.size(), maximum_batch_size_);
  size_t num_tensors = inputs_.front().input_list().size();

  // Allocate output tensors.
  std::vector<Tensor*> output_tensors(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    TensorShape shape = inputs_.front().input_list()[i].shape();
    OP_REQUIRES(
        context, shape.dim_size(0) == 1,
        errors::InvalidArgument("Batcher requires batch size 1 but was ",
                                shape.dim_size(0)));
    shape.set_dim(0, batch_size);

    OP_REQUIRES_OK(context,
                   output_list->allocate(i, shape, &output_tensors[i]));
  }

  auto work = [this, &context, &output_tensors, num_tensors](
                  int64 start, int64 end) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    for (int64 j = start; j < end; ++j) {
      for (size_t i = 0; i < num_tensors; ++i) {
        OP_REQUIRES(context,
                    inputs_[0].input_list()[i].shape() ==
                        inputs_[j].input_list()[i].shape(),
                    errors::InvalidArgument(
                        "Shapes of inputs much be equal. Shapes observed: ",
                        inputs_[0].input_list()[i].shape().DebugString(), ", ",
                        inputs_[j].input_list()[i].shape().DebugString()));

        OP_REQUIRES_OK(context,
                       tensorflow::batch_util::CopyElementToSlice(
                           inputs_[j].input_list()[i], output_tensors[i], j));
      }
    }
  };

  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  Shard(worker_threads->num_threads, worker_threads->workers, batch_size, 10,
        work);

  // New unique computation id.
  int64 new_computation_id = curr_computation_id_++;
  Tensor* computation_id_t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output("computation_id", TensorShape({}),
                                          &computation_id_t));
  computation_id_t->scalar<int64>()() = new_computation_id;

  // Move the batch of inputs into a list for the new computation.
  auto iter = std::make_move_iterator(inputs_.begin());
  being_computed_.emplace(new_computation_id,
                          std::vector<Input>{iter, iter + batch_size});
  inputs_.erase(inputs_.begin(), inputs_.begin() + batch_size);
}

void Batcher::SetOutputs(OpKernelContext* context,
                         const OpInputList& input_list, int64 computation_id) {
  mutex_lock l(mu_);
  SetOutputsInternal(context, input_list, computation_id);
  if (!context->status().ok()) {
    CancelAndClose(context);
  }
}

void Batcher::SetOutputsInternal(OpKernelContext* context,
                                 const OpInputList& input_list,
                                 int64 computation_id) {
  OP_REQUIRES(context, !is_closed_, errors::Cancelled("Batcher is closed"));

  auto search = being_computed_.find(computation_id);
  OP_REQUIRES(
      context, search != being_computed_.end(),
      errors::InvalidArgument("Invalid computation id. Id: ", computation_id));
  auto& computation_input_list = search->second;
  int64 expected_batch_size = computation_input_list.size();

  for (const Tensor& tensor : input_list) {
    OP_REQUIRES(
        context, tensor.shape().dims() > 0,
        errors::InvalidArgument(
            "Output shape must have a batch dimension. Shape observed: ",
            tensor.shape().DebugString()));
    OP_REQUIRES(
        context, tensor.shape().dim_size(0) == expected_batch_size,
        errors::InvalidArgument("Output shape must have the same batch "
                                "dimension as the input batch size. Expected: ",
                                expected_batch_size,
                                " Observed: ", tensor.shape().dim_size(0)));
  }

  auto work = [this, &input_list, &context, &computation_input_list](
                  int64 start, int64 end) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    for (int64 j = start; j < end; ++j) {
      Input& input = computation_input_list[j];

      for (size_t i = 0; i < input_list.size(); ++i) {
        TensorShape shape = input_list[i].shape();
        shape.set_dim(0, 1);

        Tensor* output_tensor;
        OP_REQUIRES_OK(context, input.context()->allocate_output(
                                    i, shape, &output_tensor));

        OP_REQUIRES_OK(context, tensorflow::batch_util::CopySliceToElement(
                                    input_list[i], output_tensor, j));
      }

      input.Done();
    }
  };

  auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
  Shard(worker_threads->num_threads, worker_threads->workers,
        expected_batch_size, 50000, work);

  being_computed_.erase(computation_id);
}

void Batcher::Close(OpKernelContext* context) {
  {
    mutex_lock l(mu_);
    CancelAndClose(context);
  }

  // Cancel all running GetInputs operations.
  full_batch_or_cancelled_cond_var_.notify_all();
}

void Batcher::CancelInput(Batcher::Input* input) {
  // Some may already have had their outputs set and the callback called so
  // they should be skipped.
  if (!input->is_valid()) {
    return;
  }

  input->context()->CtxFailure(errors::Cancelled("Compute was cancelled"));
  input->Done();
}

void Batcher::CancelAndClose(OpKernelContext* context) {
  // Something went wrong or the batcher was requested to close. All the waiting
  // Compute ops should be cancelled.

  if (is_closed_) {
    return;
  }

  for (auto& input : inputs_) {
    CancelInput(&input);
  }
  for (auto& p : being_computed_) {
    for (auto& input : p.second) {
      CancelInput(&input);
    }
  }
  is_closed_ = true;  // Causes future Compute operations to be cancelled.
}

class BatcherHandleOp : public ResourceOpKernel<Batcher> {
 public:
  explicit BatcherHandleOp(OpKernelConstruction* context)
      : ResourceOpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("minimum_batch_size", &minimum_batch_size_));
    OP_REQUIRES_OK(
        context, context->GetAttr("maximum_batch_size", &maximum_batch_size_));
    OP_REQUIRES_OK(context, context->GetAttr("timeout_ms", &timeout_ms_));
  }

 private:
  Status CreateResource(Batcher** ret) override EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    gtl::optional<std::chrono::milliseconds> timeout;
    if (timeout_ms_ != -1) {
      timeout = std::chrono::milliseconds(timeout_ms_);
    }
    *ret = new Batcher(minimum_batch_size_, maximum_batch_size_, timeout);
    return Status::OK();
  }

  int32 minimum_batch_size_;
  int32 maximum_batch_size_;
  int32 timeout_ms_;

  TF_DISALLOW_COPY_AND_ASSIGN(BatcherHandleOp);
};

class ComputeOp : public AsyncOpKernel {
 public:
  explicit ComputeOp(OpKernelConstruction* context) : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback callback) override {
    Batcher* batcher;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &batcher));

    OpInputList input_list;
    OP_REQUIRES_OK(context, context->input_list("input_list", &input_list));

    batcher->Compute(context, input_list, std::move(callback));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ComputeOp);
};

class GetInputsOp : public OpKernel {
 public:
  explicit GetInputsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Batcher* batcher;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &batcher));

    OpOutputList output_list;
    OP_REQUIRES_OK(context, context->output_list("output_list", &output_list));

    batcher->GetInputs(context, &output_list);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GetInputsOp);
};

class SetOutputsOp : public OpKernel {
 public:
  explicit SetOutputsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Batcher* batcher;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &batcher));

    OpInputList input_list;
    OP_REQUIRES_OK(context, context->input_list("input_list", &input_list));

    const Tensor* computation_id;
    OP_REQUIRES_OK(context, context->input("computation_id", &computation_id));

    batcher->SetOutputs(context, input_list, computation_id->scalar<int64>()());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SetOutputsOp);
};

class CloseOp : public OpKernel {
 public:
  explicit CloseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Batcher* batcher;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &batcher));

    batcher->Close(context);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CloseOp);
};

REGISTER_KERNEL_BUILDER(Name("Batcher").Device(DEVICE_CPU), BatcherHandleOp);

REGISTER_KERNEL_BUILDER(Name("BatcherCompute").Device(DEVICE_CPU), ComputeOp);

REGISTER_KERNEL_BUILDER(Name("BatcherGetInputs").Device(DEVICE_CPU),
                        GetInputsOp);

REGISTER_KERNEL_BUILDER(Name("BatcherSetOutputs").Device(DEVICE_CPU),
                        SetOutputsOp);

REGISTER_KERNEL_BUILDER(Name("BatcherClose").Device(DEVICE_CPU), CloseOp);

}  // namespace
}  // namespace tensorflow
