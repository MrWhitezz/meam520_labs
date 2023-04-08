import numpy as np

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
    
    def root_path(self):
        if self.parent is None:
            return [self.data]
        else:
            return self.parent.root_path() + [self.data]
    
    def traverse_for_min_distance(self, target):
        distance = np.linalg.norm(self.data - target)
        node = self
        for child in self.children:
            child_distance, child_node = child.traverse_for_min_distance(target)
            if child_distance < distance:
                distance = child_distance
                node = child_node
        return distance, node
