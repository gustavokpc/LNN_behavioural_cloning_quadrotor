import numpy as np
import pytest

from utils.data import align_state_action_targets, transform_to_sequence


def _trajectories():
    states = np.array([
        [[0, 1, 2, 3]],
        [[100, 101, 102, 103]],
    ])
    actions = np.array([
        [[10, 11, 12, 13]],
        [[110, 111, 112, 113]],
    ])
    return states, actions


def test_same_step_alignment_is_unchanged():
    states, actions = _trajectories()
    aligned_states, aligned_actions = align_state_action_targets(states, actions, "action_t")
    assert aligned_states is states
    assert aligned_actions is actions


def test_next_step_alignment_stays_inside_each_trajectory():
    states, actions = _trajectories()
    aligned_states, aligned_actions = align_state_action_targets(states, actions, "action_t+1")
    np.testing.assert_array_equal(aligned_states[:, 0], [[0, 1, 2], [100, 101, 102]])
    np.testing.assert_array_equal(aligned_actions[:, 0], [[11, 12, 13], [111, 112, 113]])


def test_next_step_alignment_uses_end_of_history_window_as_state_t():
    states, actions = _trajectories()
    windows = transform_to_sequence(states, seq_len=2)
    same_time_actions = actions[:, :, 1:]
    aligned_windows, aligned_actions = align_state_action_targets(
        windows, same_time_actions, "action_t+1"
    )
    np.testing.assert_array_equal(aligned_windows[0, 0], [[0, 1], [1, 2]])
    np.testing.assert_array_equal(aligned_actions[0, 0], [12, 13])


def test_invalid_alignment_is_rejected():
    states, actions = _trajectories()
    with pytest.raises(ValueError, match="target_alignment"):
        align_state_action_targets(states, actions, "next")
