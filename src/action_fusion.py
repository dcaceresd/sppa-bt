import numpy as np
from itertools import combinations

class ActionSegment:
    def __init__(self, start_time, end_time, gripper_state, label):
        self.start_time = start_time
        self.end_time = end_time
        self.gripper_state = gripper_state
        self.label = label

class Action:
    def __init__(self, trajectory, preconditions, postconditions, gripper_events=None, label=None, folder_path=None, segments=None):
        self.trajectory = np.array(trajectory)
        self.preconditions = set(preconditions)
        self.postconditions = set(postconditions)
        self.effect = self._compute_effect()
        self.label = label
        self.gripper_events = gripper_events
        self.folder_path = folder_path
        self.group_label = None
        self.segments = segments if segments is not None else []

    def _compute_effect(self):
        added = self.postconditions - self.preconditions
        removed = self.preconditions - self.postconditions
        return (frozenset(added), frozenset(removed))

class ActionFuser:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def group_by_effect(self, actions):
        groups = {}
        for action in actions:
            groups.setdefault(action.effect, []).append(action)
        return groups

    def _dtw_distance(self, trajectory1, trajectory2):
        n, m = len(trajectory1), len(trajectory2)
        cost = np.full((n+1, m+1), np.inf)
        cost[0, 0] = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                diff = np.linalg.norm(trajectory1[i-1] - trajectory2[j-1])
                cost[i, j] = diff + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
        return cost[n, m]

    def fuse_actions(self, actions):
        groups = self.group_by_effect(actions)
        label_counter = 0
        for effect, group in groups.items():
            base_label = f"EG_{label_counter}"
            if len(group) == 1:
                group[0].group_label = base_label
                label_counter += 1
                continue

            similar = True
            for a1, a2 in combinations(group, 2):
                if self._dtw_distance(a1.trajectory, a2.trajectory) > self.epsilon:
                    similar = False
                    break

            if similar:
                for action in group:
                    action.group_label = base_label
                label_counter += 1
            else:
                for idx, action in enumerate(group):
                    action.group_label = f"{base_label}_{idx}"
                label_counter += 1

        return actions

# Modified test case with string conditions
# action1 = Action(
#     trajectory=[[0, 0, 0], [1, 1, 0.5], [2, 2, 1]],
#     preconditions={'x=0', 'y=0', 'z=1', 'q=0'},
#     postconditions={'x=1', 'y=0', 'z=1', 'q=0'}  # Added: x=1, Removed: x=0
# )

# action2 = Action(
#     trajectory=[[0, 0, 0], [0.9, 1.1, 0.6], [2.1, 1.9, 1.1]],
#     preconditions={'x=0', 'y=0', 'z=1', 'q=0'},
#     postconditions={'x=1', 'y=1', 'z=1', 'q=0'}  # Added: x=1,y=1, Removed: x=0,y=0
# )

# action3 = Action(
#     trajectory=[[0, 0, 0], [0, 0, 1], [0, 0, 2]],
#     preconditions={'x=0', 'y=0', 'z=1', 'q=0'},
#     postconditions={'x=1', 'y=0', 'z=1', 'q=0'}  # Same effect as action1
# )

# action4 = Action(
#     trajectory=[[0, 0, 0], [0, 0, 1.2], [0, 0, 1.8]],
#     preconditions={'x=0', 'y=0', 'z=1', 'q=0'},
#     postconditions={'y=0', 'z=2', 'q=0'}  # Different effect: Added z=2, Removed x=0,z=1
# )

# # Create fuser with threshold ε=1.5
# fuser = ActionFuser(epsilon=1.5)
# actions = [action1, action2, action3, action4]
# fused_actions = fuser.fuse_actions(actions)

# # Print results
# for action in fused_actions:
#     added_str = ', '.join(str(c) for c in action.effect[0]) if action.effect[0] else '∅'
#     removed_str = ', '.join(str(c) for c in action.effect[1]) if action.effect[1] else '∅'
#     print(f"Trajectory shape: {action.trajectory.shape}")
#     print(f"Effect: (+{added_str} | -{removed_str})")
#     print(f"Label: {action.label}\n")