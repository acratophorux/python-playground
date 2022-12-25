###   This code defines a decision tree classifier in Python using the NumPy library. The decision tree is a supervised machine learning algorithm that can be used for classification or regression tasks. It works by building a tree-like model of decisions based on the features of the input data.

###   The code consists of two main classes: Node and DecisionTree. The Node class represents a node in the decision tree, and it has several attributes:

#       feature: the feature used to split the data at this node
#       threshold: the threshold used to split the data at this node
#       left: a reference to the left child node
#       right: a reference to the right child node
#       value: the predicted value of this node, if it is a leaf node

###   The DecisionTree class represents the decision tree model, and it has several attributes and methods:

#       max_depth: the maximum depth of the tree (i.e., the maximum number of nodes from the root to a leaf)
#       min_samples_split: the minimum number of samples required to split a node
#       root: a reference to the root node of the tree
#       _is_finished(): a helper method that determines whether the tree building process should stop at the current node
#       _build_tree(): a recursive method that builds the tree by finding the best split at each node and growing the children nodes
#       fit(): a method that trains the decision tree model on a given dataset
#       _traverse_tree(): a recursive method that traverses the tree and returns the predicted value for a given input sample
#       predict(): a method that predicts the output for a given input dataset
#       _entropy(): a helper method that calculates the entropy of a given target variable
#       _create_split(): a helper method that splits a dataset based on a given threshold
#       _information_gain(): a helper method that calculates the information gain of a given split
#       _best_split(): a helper method that finds the best split for a given feature

###   The fit() method is used to train the decision tree model on a given dataset. It starts by calling the _build_tree() method, which builds the tree recursively. At each node, the method finds the best split by iterating over the features and threshold values and selecting the split that maximizes the information gain. The process continues until the stopping criteria are met (e.g., maximum depth, minimum number of samples, etc.).

###   The predict() method is used to predict the output for a given input dataset. It calls the _traverse_tree() method for each input sample, which traverses the tree and returns the predicted value for that sample.



import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def _is_finished(self, depth):
        if(depth >= self.max_depth or self.n_class_labels == 1 or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)
        
        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        
        return Node(best_feat, best_thresh, left_child, right_child)

    def fit(self, X, y):
        self.root = self._build_tree(X, y) # root will be populated last
    
    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x, node : Node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def _create_split(self, X, thresh):
        # X is a feature column
		# returns two arrays of indices: one for rows where `X <= thresh` and one for rows where `X <= thresh`
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere( X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0
        
        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {'score':-1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh
        
        return split['feat'], split['thresh']

