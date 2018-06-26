# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for DMLab-30."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf


LEVEL_MAPPING = collections.OrderedDict([
    ('rooms_collect_good_objects_train', 'rooms_collect_good_objects_test'),
    ('rooms_exploit_deferred_effects_train',
     'rooms_exploit_deferred_effects_test'),
    ('rooms_select_nonmatching_object', 'rooms_select_nonmatching_object'),
    ('rooms_watermaze', 'rooms_watermaze'),
    ('rooms_keys_doors_puzzle', 'rooms_keys_doors_puzzle'),
    ('language_select_described_object', 'language_select_described_object'),
    ('language_select_located_object', 'language_select_located_object'),
    ('language_execute_random_task', 'language_execute_random_task'),
    ('language_answer_quantitative_question',
     'language_answer_quantitative_question'),
    ('lasertag_one_opponent_small', 'lasertag_one_opponent_small'),
    ('lasertag_three_opponents_small', 'lasertag_three_opponents_small'),
    ('lasertag_one_opponent_large', 'lasertag_one_opponent_large'),
    ('lasertag_three_opponents_large', 'lasertag_three_opponents_large'),
    ('natlab_fixed_large_map', 'natlab_fixed_large_map'),
    ('natlab_varying_map_regrowth', 'natlab_varying_map_regrowth'),
    ('natlab_varying_map_randomized', 'natlab_varying_map_randomized'),
    ('skymaze_irreversible_path_hard', 'skymaze_irreversible_path_hard'),
    ('skymaze_irreversible_path_varied', 'skymaze_irreversible_path_varied'),
    ('psychlab_arbitrary_visuomotor_mapping',
     'psychlab_arbitrary_visuomotor_mapping'),
    ('psychlab_continuous_recognition', 'psychlab_continuous_recognition'),
    ('psychlab_sequential_comparison', 'psychlab_sequential_comparison'),
    ('psychlab_visual_search', 'psychlab_visual_search'),
    ('explore_object_locations_small', 'explore_object_locations_small'),
    ('explore_object_locations_large', 'explore_object_locations_large'),
    ('explore_obstructed_goals_small', 'explore_obstructed_goals_small'),
    ('explore_obstructed_goals_large', 'explore_obstructed_goals_large'),
    ('explore_goal_locations_small', 'explore_goal_locations_small'),
    ('explore_goal_locations_large', 'explore_goal_locations_large'),
    ('explore_object_rewards_few', 'explore_object_rewards_few'),
    ('explore_object_rewards_many', 'explore_object_rewards_many'),
])

HUMAN_SCORES = {
    'rooms_collect_good_objects_test': 10,
    'rooms_exploit_deferred_effects_test': 85.65,
    'rooms_select_nonmatching_object': 65.9,
    'rooms_watermaze': 54,
    'rooms_keys_doors_puzzle': 53.8,
    'language_select_described_object': 389.5,
    'language_select_located_object': 280.7,
    'language_execute_random_task': 254.05,
    'language_answer_quantitative_question': 184.5,
    'lasertag_one_opponent_small': 12.65,
    'lasertag_three_opponents_small': 18.55,
    'lasertag_one_opponent_large': 18.6,
    'lasertag_three_opponents_large': 31.5,
    'natlab_fixed_large_map': 36.9,
    'natlab_varying_map_regrowth': 24.45,
    'natlab_varying_map_randomized': 42.35,
    'skymaze_irreversible_path_hard': 100,
    'skymaze_irreversible_path_varied': 100,
    'psychlab_arbitrary_visuomotor_mapping': 58.75,
    'psychlab_continuous_recognition': 58.3,
    'psychlab_sequential_comparison': 39.5,
    'psychlab_visual_search': 78.5,
    'explore_object_locations_small': 74.45,
    'explore_object_locations_large': 65.65,
    'explore_obstructed_goals_small': 206,
    'explore_obstructed_goals_large': 119.5,
    'explore_goal_locations_small': 267.5,
    'explore_goal_locations_large': 194.5,
    'explore_object_rewards_few': 77.7,
    'explore_object_rewards_many': 106.7,
}

RANDOM_SCORES = {
    'rooms_collect_good_objects_test': 0.073,
    'rooms_exploit_deferred_effects_test': 8.501,
    'rooms_select_nonmatching_object': 0.312,
    'rooms_watermaze': 4.065,
    'rooms_keys_doors_puzzle': 4.135,
    'language_select_described_object': -0.07,
    'language_select_located_object': 1.929,
    'language_execute_random_task': -5.913,
    'language_answer_quantitative_question': -0.33,
    'lasertag_one_opponent_small': -0.224,
    'lasertag_three_opponents_small': -0.214,
    'lasertag_one_opponent_large': -0.083,
    'lasertag_three_opponents_large': -0.102,
    'natlab_fixed_large_map': 2.173,
    'natlab_varying_map_regrowth': 2.989,
    'natlab_varying_map_randomized': 7.346,
    'skymaze_irreversible_path_hard': 0.1,
    'skymaze_irreversible_path_varied': 14.4,
    'psychlab_arbitrary_visuomotor_mapping': 0.163,
    'psychlab_continuous_recognition': 0.224,
    'psychlab_sequential_comparison': 0.129,
    'psychlab_visual_search': 0.085,
    'explore_object_locations_small': 3.575,
    'explore_object_locations_large': 4.673,
    'explore_obstructed_goals_small': 6.76,
    'explore_obstructed_goals_large': 2.61,
    'explore_goal_locations_small': 7.66,
    'explore_goal_locations_large': 3.14,
    'explore_object_rewards_few': 2.073,
    'explore_object_rewards_many': 2.438,
}

ALL_LEVELS = frozenset([
    'rooms_collect_good_objects_train',
    'rooms_collect_good_objects_test',
    'rooms_exploit_deferred_effects_train',
    'rooms_exploit_deferred_effects_test',
    'rooms_select_nonmatching_object',
    'rooms_watermaze',
    'rooms_keys_doors_puzzle',
    'language_select_described_object',
    'language_select_located_object',
    'language_execute_random_task',
    'language_answer_quantitative_question',
    'lasertag_one_opponent_small',
    'lasertag_three_opponents_small',
    'lasertag_one_opponent_large',
    'lasertag_three_opponents_large',
    'natlab_fixed_large_map',
    'natlab_varying_map_regrowth',
    'natlab_varying_map_randomized',
    'skymaze_irreversible_path_hard',
    'skymaze_irreversible_path_varied',
    'psychlab_arbitrary_visuomotor_mapping',
    'psychlab_continuous_recognition',
    'psychlab_sequential_comparison',
    'psychlab_visual_search',
    'explore_object_locations_small',
    'explore_object_locations_large',
    'explore_obstructed_goals_small',
    'explore_obstructed_goals_large',
    'explore_goal_locations_small',
    'explore_goal_locations_large',
    'explore_object_rewards_few',
    'explore_object_rewards_many',
])


def _transform_level_returns(level_returns):
  """Converts training level names to test level names."""
  new_level_returns = {}
  for level_name, returns in level_returns.iteritems():
    new_level_returns[LEVEL_MAPPING.get(level_name, level_name)] = returns

  test_set = set(LEVEL_MAPPING.values())
  diff = test_set - set(new_level_returns.keys())
  if diff:
    raise ValueError('Missing levels: %s' % list(diff))

  for level_name, returns in new_level_returns.iteritems():
    if level_name in test_set:
      if not returns:
        raise ValueError('Missing returns for level: \'%s\': ' % level_name)
    else:
      tf.logging.info('Skipping level %s for calculation.', level_name)

  return new_level_returns


def compute_human_normalized_score(level_returns, per_level_cap):
  """Computes human normalized score.

  Levels that have different training and test versions, will use the returns
  for the training level to calculate the score. E.g.
  'rooms_collect_good_objects_train' will be used for
  'rooms_collect_good_objects_test'. All returns for levels not in DmLab-30
  will be ignored.

  Args:
    level_returns: A dictionary from level to list of episode returns.
    per_level_cap: A percentage cap (e.g. 100.) on the per level human
      normalized score. If None, no cap is applied.

  Returns:
    A float with the human normalized score in percentage.

  Raises:
    ValueError: If a level is missing from `level_returns` or has no returns.
  """
  new_level_returns = _transform_level_returns(level_returns)

  def human_normalized_score(level_name, returns):
    score = np.mean(returns)
    human = HUMAN_SCORES[level_name]
    random = RANDOM_SCORES[level_name]
    human_normalized_score = (score - random) / (human - random) * 100
    if per_level_cap is not None:
      human_normalized_score = min(human_normalized_score, per_level_cap)
    return human_normalized_score

  return np.mean(
      [human_normalized_score(k, v) for k, v in new_level_returns.items()])
