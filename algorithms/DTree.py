import pandas as pd


class Node:
    def __init__(self, value, name=None):
        self.name = name
        self.value = value
        self.parent = None
        self.children = []
        self.children_num = 0
        self.data = None
        self.loss = 0

    def load_data(self, data_path):
        self.data = pd.read_csv(data_path)

    def division(self):
        pass


class DTree:
    def __init__(self, root):
        self.root = root

    def fit(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    root = Node(0, "root")
    root.load_data()
    tree = DTree(root)
