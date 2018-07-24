# Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

This repository contains an implementation of "Importance Weighted Actor-Learner
Architectures", along with a *dynamic batching* module. This is not an
officially supported Google product.

For a detailed description of the architecture please read [our paper][arxiv].
Please cite the paper if you use the code from this repository in your work.

### Bibtex

```
@inproceedings{impala2018,
  title={IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures},
  author={Espeholt, Lasse and Soyer, Hubert and Munos, Remi and Simonyan, Karen and Mnih, Volodymir and Ward, Tom and Doron, Yotam and Firoiu, Vlad and Harley, Tim and Dunning, Iain and others},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2018}
}
```

## Running the Code

### Prerequisites

[TensorFlow][tensorflow] >=1.9.0-dev20180530, the environment
[DeepMind Lab][deepmind_lab] and the neural network library
[DeepMind Sonnet][sonnet]. Although we use [DeepMind Lab][deepmind_lab] in this
release, the agent has been successfully applied to other domains such as
[Atari][arxiv], [Street View][learning_nav] and has been modified to
[generate images][generate_images].

We include a [Dockerfile][dockerfile] that serves as a reference for the
prerequisites and commands needed to run the code.

### Single Machine Training on a Single Level

Training on `explore_goal_locations_small`. Most runs should end up with average
episode returns around 200 or around 250 after 1B frames.

```sh
python experiment.py --num_actors=48 --batch_size=32
```

Adjust the number of actors (i.e. number of environments) and batch size to
match the size of the machine it runs on. A single actor, including DeepMind
Lab, requires a few hundred MB of RAM.

### Distributed Training on DMLab-30

Training on the full [DMLab-30][dmlab30]. Across 10 runs with different seeds
but identical hyperparameters, we observed between 45 and 50 capped human
normalized training score with different seeds (`--seed=[seed]`). Test scores
are usually an absolute of ~2% lower.

#### Learner

```sh
python experiment.py --job_name=learner --task=0 --num_actors=150 \
    --level_name=dmlab30 --batch_size=32 --entropy_cost=0.0033391318945337044 \
    --learning_rate=0.00031866995608948655 \
    --total_environment_frames=10000000000 --reward_clipping=soft_asymmetric
```

#### Actor(s)

```sh
for i in $(seq 0 149); do
  python experiment.py --job_name=actor --task=$i \
      --num_actors=150 --level_name=dmlab30 --dataset_path=[...] &
done;
wait
```

#### Test Score

```sh
python experiment.py --mode=test --level_name=dmlab30 --dataset_path=[...] \
    --test_num_episodes=10
```

[arxiv]: https://arxiv.org/abs/1802.01561
[deepmind_lab]: https://github.com/deepmind/lab
[sonnet]: https://github.com/deepmind/sonnet
[learning_nav]: https://arxiv.org/abs/1804.00168
[generate_images]: https://deepmind.com/blog/learning-to-generate-images/
[tensorflow]: https://github.com/tensorflow/tensorflow
[dockerfile]: Dockerfile
[dmlab30]: https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30
