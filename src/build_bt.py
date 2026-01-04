import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import sys

sys.setrecursionlimit(15000)
print("New recursion limit:", sys.getrecursionlimit())

def build_behavior_tree_from_multiple_demos(all_demonstrations):
    """
    Build a behavior tree from multiple demonstrations.
    
    Args:
        all_demonstrations (list): List of demonstrations, where each demonstration
                                  is a list of (action, post_conditions, pre_conditions) tuples.
    
    Returns:
        ET.Element: The root element of the behavior tree.
    """
    reference_demo = find_reference_demonstration(all_demonstrations)
    action_order = determine_action_order(all_demonstrations, reference_demo)
    actions_with_common_conditions = extract_common_conditions(all_demonstrations, action_order)
    
    return build_behavior_tree_recursive(actions_with_common_conditions)

def find_reference_demonstration(all_demonstrations):
    """
    Find the demonstration with the least number of pre- and post-conditions.
    
    Args:
        all_demonstrations (list): List of demonstrations.
    
    Returns:
        list: The reference demonstration.
    """
    min_conditions = float('inf')
    reference_demo = None
    
    for demo in all_demonstrations:
        total_conditions = sum(
            len(pre_conditions) if isinstance(pre_conditions, (list, set)) else 1
            + len(post_conditions) if isinstance(post_conditions, (list, set)) else 1
            for _, post_conditions, pre_conditions in demo
        )
        
        if total_conditions < min_conditions:
            min_conditions = total_conditions
            reference_demo = demo
    
    return reference_demo

def determine_action_order(all_demonstrations, reference_demo):
    """
    Determine the most common order of actions across all demonstrations.
    
    Args:
        all_demonstrations (list): List of demonstrations.
        reference_demo (list): The reference demonstration.
    
    Returns:
        list: Ordered list of action names.
    """
    action_sequences = []
    for demo in all_demonstrations:
        action_sequences.append([action for action, _, _ in demo])
    
    action_pairs = Counter()
    for sequence in action_sequences:
        for i in range(len(sequence) - 1):
            action_pairs[(sequence[i], sequence[i+1])] += 1
    
    ordered_actions = []
    remaining_actions = set(action for demo in all_demonstrations for action, _, _ in demo)
    
    first_actions = Counter(seq[0] for seq in action_sequences if seq)
    if first_actions:
        most_common_first = first_actions.most_common(1)[0][0]
        ordered_actions.append(most_common_first)
        remaining_actions.remove(most_common_first)
    
    while remaining_actions and len(ordered_actions) < len(remaining_actions) + 1:
        current = ordered_actions[-1]
        next_candidates = [(next_action, count) for (curr, next_action), count in action_pairs.items() 
                          if curr == current and next_action in remaining_actions]
        
        if not next_candidates:
            break
            
        next_action = max(next_candidates, key=lambda x: x[1])[0]
        ordered_actions.append(next_action)
        remaining_actions.remove(next_action)
    
    if remaining_actions:
        reference_actions = [action for action, _, _ in reference_demo]
        final_order = ordered_actions.copy()
        
        for action in reference_actions:
            if action not in final_order and action in remaining_actions:
                final_order.append(action)
                remaining_actions.remove(action)
        
        final_order.extend(remaining_actions)
        return final_order
    
    return ordered_actions

def extract_common_conditions(all_demonstrations, action_order):
    """
    Extract common pre- and post-conditions for each action across demonstrations.
    
    Args:
        all_demonstrations (list): List of demonstrations.
        action_order (list): Ordered list of action names.
    
    Returns:
        list: List of (action, common_post_conditions, common_pre_conditions) tuples.
    """
    action_conditions = defaultdict(lambda: {'pre': [], 'post': []})
    
    for demo in all_demonstrations:
        for action, post_conditions, pre_conditions in demo:
            pre_set = set(pre_conditions) if isinstance(pre_conditions, (list, set)) else {pre_conditions}
            post_set = set(post_conditions) if isinstance(post_conditions, (list, set)) else {post_conditions}
            
            action_conditions[action]['pre'].append(pre_set)
            action_conditions[action]['post'].append(post_set)
    
    common_conditions = []
    for action in action_order:
        if action in action_conditions:
            if action_conditions[action]['pre']:
                common_pre = set.intersection(*action_conditions[action]['pre'])
            else:
                common_pre = set()

            if action_conditions[action]['post']:
                common_post = set.intersection(*action_conditions[action]['post'])
            else:
                common_post = set()
            if len(common_pre) == 1:
                common_pre = list(common_pre)[0]
            elif len(common_pre) > 1:
                common_pre = list(common_pre)

            if len(common_post) == 1:
                common_post = list(common_post)[0]
            elif len(common_post) > 1:
                common_post = list(common_post)

            common_conditions.append((action, common_post, common_pre))

    return common_conditions


def build_behavior_tree_recursive(actions_list, root=None, tree_index=1, current_elem=None):
    actions_list = actions_list[::-1]
    if root is None:
        root = ET.Element("root", attrib={"main_tree_to_execute": "MainTree"})
        main_tree = ET.SubElement(root, "BehaviorTree", attrib={"ID": "MainTree"})
        tree_connector = ET.SubElement(main_tree, "Fallback")
        behavior_tree = ET.SubElement(root, "BehaviorTree", attrib={"ID": f"SubTree-{tree_index}"})
        fallback = ET.SubElement(behavior_tree, "Fallback")
        current_elem = fallback
    else:
        tree_connector = root.find("./BehaviorTree[@ID='MainTree']/Fallback")

    def process_action(index, tree_index, current_elem):
        if index >= len(actions_list):
            return tree_index  

        action, post_conditions, pre_conditions = actions_list[index]

        if index > 0 and not conditions_match(actions_list[index - 1][2], post_conditions):
            ET.SubElement(tree_connector, "SubTree", attrib={"ID": f"SubTree-{tree_index}"})
            
            tree_index += 1
            behavior_tree = ET.SubElement(root, "BehaviorTree", attrib={"ID": f"SubTree-{tree_index}"})
            fallback = ET.SubElement(behavior_tree, "Fallback")
            current_elem = fallback
            add_conditions(current_elem, post_conditions)
        
        if index == 0:
            add_conditions(current_elem, post_conditions)
        
        sequence = ET.SubElement(current_elem, "Sequence")

        if index < len(actions_list) - 1:  
            fallback_next = ET.SubElement(sequence, "Fallback")
            add_conditions(fallback_next, pre_conditions)
            ET.SubElement(sequence, "Action", attrib={"ID": "DMP", "dmp_name": action})
            return process_action(index + 1, tree_index, fallback_next)
        else: 
            add_conditions(sequence, pre_conditions)
            ET.SubElement(sequence, "Action", attrib={"ID": "DMP", "dmp_name": action})
            ET.SubElement(tree_connector, "SubTree", attrib={"ID": f"SubTree-{tree_index}"})
            return tree_index

    # Start the recursive process
    tree_index = process_action(0, tree_index, current_elem)

    # Final cleanup
    restructure_behavior_tree_with_common_conditions(root)
    simplify_fallback_sequence_nodes(root)
    remove_unnecessary_control_nodes(root)
    print("Removing empty control nodes")
    remove_empty_control_nodes(root)
    remove_unnecessary_control_nodes(root)
    return root


def conditions_match(prev_post_conditions, current_pre_conditions):
    """
    Check if previous post conditions match the current pre conditions.

    Args:
        prev_post_conditions (list or str): The post conditions from the previous action.
        current_pre_conditions (list or str): The pre conditions of the current action.
    
    Returns:
        bool: True if the conditions match, False otherwise.
    """

    if isinstance(prev_post_conditions, (list, set)) and isinstance(current_pre_conditions, (list, set)):
        return set(prev_post_conditions) == set(current_pre_conditions)
    elif isinstance(prev_post_conditions, (list, set)):
        return current_pre_conditions in prev_post_conditions
    elif isinstance(current_pre_conditions, (list, set)):
        return prev_post_conditions in current_pre_conditions
    else:
        return prev_post_conditions == current_pre_conditions

def add_conditions(parent, conditions):
    if isinstance(conditions, (list, set)):
        sequence = ET.SubElement(parent, "Sequence")
        for condition in conditions:
            ET.SubElement(sequence, "Condition", attrib={"ID": "VLMCondition", "condition_name": condition})
    else:
        ET.SubElement(parent, "Condition", attrib={"ID": "VLMCondition", "condition_name": conditions})

def remove_empty_control_nodes(root):
    """
    Removes control nodes that do not have any children from the behavior tree.
    """
    # Iterate over all elements in the tree
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ["Sequence", "Fallback", "Parallel", "Decorator"]:
                if len(list(child)) == 0:
                    parent.remove(child)

    return root

def remove_duplicate_conditions(root):
    def remove_duplicates(parent):
        seen_conditions = set()
        for elem in list(parent):
            if elem.tag == "Sequence":
                conditions_ids = tuple(child.attrib.get("condition_name") for child in elem if child.tag == "Condition")
                if conditions_ids in seen_conditions:
                    parent.remove(elem)
                else:
                    seen_conditions.add(conditions_ids)
            elif elem.tag in ("Fallback", "Sequence"):
                remove_duplicates(elem)

    for elem in root.iter():
        if elem.tag in ("Fallback", "Sequence"):
            remove_duplicates(elem)

def simplify_fallback_sequence_nodes(root):
    def simplify_node(node):
        for child in list(node):
            simplify_node(child)

            if node.tag == "Sequence" and child.tag == "Sequence":
                if all(grandchild.tag == "Condition" for grandchild in child):
                    for grandchild in list(child):
                        node.insert(list(node).index(child), grandchild)
                    node.remove(child)
            elif child.tag == "Fallback" and len(child) == 1 and child[0].tag == "Sequence":
                single_sequence = child[0]
                node.insert(list(node).index(child), single_sequence)
                node.remove(child)
            elif child.tag == "Sequence" and len(child) == 1 and child[0].tag == "Fallback":
                single_fallback = child[0]
                node.insert(list(node).index(child), single_fallback)
                node.remove(child)

    simplify_node(root)

def remove_unnecessary_control_nodes(root):
    """
    Helper function to simplify control nodes with only one child condition or action
    """
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "Fallback" and len(child) == 1 and (child[0].tag == "Condition" or child[0].tag == "Action"):
                node = child[0]
                parent.insert(list(parent).index(child), node)
                parent.remove(child)
            if child.tag == "Sequence" and len(child) == 1 and (child[0].tag == "Condition" or child[0].tag == "Action"):
                node = child[0]
                parent.insert(list(parent).index(child), node)
                parent.remove(child)


def find_common_conditions_in_subtrees(root):
    """
    Identifies conditions that are common across all Sequence or Fallback nodes in each subtree.
    Ignores Sequence or Fallback nodes that do not contain any Condition elements.
    Only conditions that repeat in every relevant Sequence or Fallback node within a subtree are identified.
    """
    subtree_common_conditions = {}

    # Iterate over each subtree
    for behavior_tree in root.findall("BehaviorTree"):
        subtree_id = behavior_tree.attrib.get("ID")
        condition_counts = defaultdict(int)
        relevant_node_count = 0

        for node in behavior_tree.iter():
            if node.tag in {"Sequence", "Fallback"}:
                # Collect all condition names in this node
                conditions_in_node = {
                    child.attrib.get("condition_name") for child in node if child.tag == "Condition"
                }

                if not conditions_in_node:
                    continue

                relevant_node_count += 1

                for condition_id in conditions_in_node:
                    condition_counts[condition_id] += 1

        # Identify conditions that appear in all relevant Sequence or Fallback nodes in this subtree
        common_conditions = {
            condition_id for condition_id, count in condition_counts.items() if count == relevant_node_count
        }

        if common_conditions:
            subtree_common_conditions[subtree_id] = common_conditions

    return subtree_common_conditions

def restructure_behavior_tree_with_common_conditions(root):
    """
    Moves common conditions identified by find_common_conditions_in_subtrees to the top
    of each subtree, and removes duplicates within the subtree.
    """

    common_conditions_per_subtree = find_common_conditions_in_subtrees(root)

    # Iterate over each subtree
    for behavior_tree in root.findall("BehaviorTree"):
        subtree_id = behavior_tree.attrib.get("ID")
        print(f"Processing subtree: {subtree_id}")
        
        common_conditions = common_conditions_per_subtree.get(subtree_id, set())
        
        if not common_conditions:
            # Skip subtrees without common conditions
            continue

        # Create a new Sequence node at the top with the common conditions
        common_conditions_sequence = ET.Element("SequenceStar")
        for condition_id in common_conditions:
            condition = ET.Element("Condition", ID="VLMCondition", condition_name=condition_id)
            common_conditions_sequence.append(condition)

        def remove_common_conditions(node):
            to_remove = []
            for child in node:
                if child.tag == "Condition" and child.attrib.get("condition_name") in common_conditions:
                    to_remove.append(child)
                elif child.tag in {"Sequence", "Fallback"}:
                    remove_common_conditions(child)

            for child in to_remove:
                node.remove(child)

        remove_common_conditions(behavior_tree)

        modified_subtree = list(behavior_tree)  
        common_conditions_sequence.extend(modified_subtree)
        print(common_conditions_sequence)

        behavior_tree.clear()  
        behavior_tree.attrib["ID"] = subtree_id  
        behavior_tree.append(common_conditions_sequence)  


def save_tree_to_file(root, filename="behavior_tree.xml"):
    tree = ET.ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)


# Example usage
if __name__ == "__main__":
    # Example of multiple demonstrations
    demo1 = [
        ['a_1', ['c_1', 'c_x'], ['c_0', 'c_x']],
        ['a_2', ['c_2_1', 'c_2_2', 'c_x'], ['c_1', 'c_x']],
        ['a_3', ['c_x', 'c_3'], ['c_x', 'c_2_1', 'c_2_2']],
        ['a_4', 'c_4', ['c_3_1', 'c_3_2']]
    ]
    demo1 = demo1[::-1]
    demo2 = [
        ['a_1', ['c_1', 'c_x'], ['c_0', 'c_x']],
        ['a_2', ['c_2_1', 'c_x'], ['c_1', 'c_x']],
        ['a_3', ['c_x', 'c_3'], ['c_x', 'c_2_1']],
        ['a_4', 'c_4', ['c_3_1', 'c_3_2']]
    ]
    demo2 = demo2[::-1]
    
    demo3 = [
        ['a_1', ['c_1', 'c_x'], ['c_0', 'c_x']],
        ['a_2', ['c_2_1', 'c_2_2', 'c_x'], ['c_1', 'c_x']],
        ['a_3', ['c_x', 'c_3'], ['c_x', 'c_2_1', 'c_2_2']],
        ['a_4', 'c_4', ['c_3_1', 'c_3_2']]
    ]
    demo3 = demo3[::-1]
    all_demonstrations = [demo1, demo2, demo3]
    
    # Build behavior tree from multiple demonstrations
    root = build_behavior_tree_from_multiple_demos(all_demonstrations)
    save_tree_to_file(root, filename="multi_demo_behavior_tree.xml")