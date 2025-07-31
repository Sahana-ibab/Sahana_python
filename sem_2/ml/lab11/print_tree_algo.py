# Function to visualize the tree
def print_tree(node, depth=0):
    if node.value is not None:
        print(f"{'  ' * depth}Leaf: Class {node.value}")
    else:
        print(f"{'  ' * depth}Feature {node.feature} <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)