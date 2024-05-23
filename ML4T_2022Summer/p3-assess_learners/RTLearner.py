import numpy as np


class RTLearner(object):
    """
    This is a Random Tree Learner. Contains the code for the regression Random Tree class
    Implemented correctly? Who knows!

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, leaf_size=0, verbose=False):
        """
        Constructor method
        """
        if verbose:
            print("RTLearner leaf_size: ", leaf_size)
            if leaf_size <= 0:
                print("Leaf size <= 0: ", leaf_size, "; good luck with that!")
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "dstrube3"

    def build_tree(self, data):
        # If tree is empty or if all data.y is the same
        if data.shape[0] <= self.leaf_size or np.unique(data[:, -1]).shape[0] == 1:
            return np.array([[-1, data[0][-1], None, None]])

        # Else, determine best feature to split on
        # Best feature to split on: random
        best_feature = np.random.randint(data[:, 0:-1].shape[1])

        # Splitting on median to keep the tree balanced is boring.
        # Let's keep this crazy train (medicated by a BagLearner) rolling!
        random_1 = np.random.randint(data.shape[0])
        random_2 = np.random.randint(data.shape[0])
        split_val = (data[random_1, best_feature] + data[random_2, best_feature]) / 2

        if np.all(data[:, best_feature] <= split_val):
            return np.array([[-1, np.mean(data[:, -1]), None, None]])

        left_tree = self.build_tree(data[data[:, best_feature] <= split_val])
        right_tree = self.build_tree(data[data[:, best_feature] > split_val])
        root = np.array(([best_feature, split_val, 1, left_tree.shape[0] + 1],))
        return np.concatenate((root, left_tree, right_tree), axis=0)

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        new_data = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data[:, 0: data_x.shape[1]] = data_x
        new_data[:, -1] = data_y

        # build and save the model
        self.tree = self.build_tree(new_data)

        if self.verbose:
            print("new_data:\n", new_data)
            print("new_data shape:\n", new_data.shape)
            print("tree:\n", self.build_tree(new_data))
            print("tree shape:\n", self.build_tree(new_data).shape)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predicted_result = np.array(())
        for tree_row in points:
            tree_node = 0
            while int(self.tree[tree_node, 0]) != -1:
                index = int(self.tree[tree_node, 0])

                if tree_row[index] <= self.tree[tree_node, 1]:
                    tree_node += int(self.tree[tree_node, 2])
                else:
                    tree_node += + int(self.tree[tree_node, 3])

            temp_y = self.tree[tree_node, 1]
            predicted_result = np.append(predicted_result, np.array(temp_y))
        return predicted_result


if __name__ == "__main__":
    # If verbose = False your code must not generate ANY output
    pass
