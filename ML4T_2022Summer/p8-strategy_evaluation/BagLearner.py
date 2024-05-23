# TODO Adjust leaf_size.
# TODO Adjust the number of bags.
import numpy as np
from RTLearner import RTLearner


class BagLearner(object):
    # From P3
    """
    This is a Bootstrap Aggregating Learner (i.e, a “bag learner”).
    Contains the code for the classification Bag Learner (i.e., a BagLearner containing Random Trees)

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, learner=RTLearner, kwargs=None, bags=20, boost=False, verbose=False):
        """
        Constructor method
        """
        if verbose:
            print("BagLearner learner: ", learner, ", kwargs: ", kwargs, ", bags: ", bags, ", boost: ", boost)
            if kwargs is None:
                print("kwargs is None.")
            if bags <= 0:
                print("bags is <= 0. Good luck!")
            if learner is None:
                print("Learner type is None. Are you sure about that?")
            elif learner is not RTLearner and learner is not BagLearner:
                # ArbitraryLearner: parse all kwargs into whatever
                print("Unexpected learner type: ", learner, "; hope this works...")
            if boost:
                print("Sorry, boost is not implemented yet.")

        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []

        for i in range(0, self.bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return 'dstrube3'

    def add_evidence(self, data_x, data_y):
        # Same logic for regression or classification:
        new_data = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data[:, 0: data_x.shape[1]] = data_x
        new_data[:, -1] = data_y
        for i in range(0, self.bags):
            index = np.random.randint(0, new_data.shape[0], new_data.shape[0])
            data_x_i = new_data[index, :-1]
            data_y_i = new_data[index, -1]
            self.learners[i].add_evidence(data_x_i, data_y_i)

    def query(self, points):
        # For regression:
        # query_results = []
        # for learner in self.learners:
        #     query_result = learner.query(points)
        #     query_results.append(query_result)
        # query_results_npa = np.array(query_results)
        # query_results_mean = np.mean(query_results_npa, axis=0)
        # return query_results_mean

        # For classification:
        predicted_classifications = np.ones([self.bags, points.shape[0]])
        for i in range(self.bags):
            predicted_classifications[i] = self.learners[i].query(points)
        # For classification, you must convert your regression learner to use mode rather than mean
        # Numpy has a .mean() but no .mode()???
        uniques, counts = np.unique(predicted_classifications, return_counts=True)
        max_count = np.argmax(counts)
        return uniques[max_count]


if __name__ == "__main__":
    # If verbose = False your code must not generate ANY output
    pass
