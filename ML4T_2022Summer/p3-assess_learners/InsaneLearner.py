
"""
# My second attempt:
import numpy as np, BagLearner as bl, LinRegLearner as lrl
class InsaneLearner(bl):
    def __init__(self, verbose=False): self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, bags=20, kwargs={}, boost=False, verbose=False) for _ in range(0, 20)]
    def author(self): return "dstrube3"
# Result:
# TypeError: module.__init__() takes at most 2 arguments (3 given)
# ???
"""

"""
# Improvements from Ryan K Nah :
class InsaneLearner(__import__("BagLearner").BagLearner):
    def __init__(self, verbose: bool): super().__init__(__import__("BagLearner").BagLearner, kwargs={"learner": __import__("LinRegLearner").LinRegLearner, "kwargs": {}, "bags": 20, "boost": False, "verbose": verbose}, bags=20, boost=False, verbose=verbose)
    def author(self): return "dstrube3"
# inlined imports, & no need for add_evidence or query since they are not explicitly required (and they are inherited):
# "Hint:  Only include methods necessary to run the assignment tasks and the author methods.  "
# https://lucylabs.gatech.edu/ml4t/summer2022/project-3/
# BUT ^Gets error: 
# class InsaneLearner(__import__("BagLearner").BagLearner):
# AttributeError: module 'BagLearner' has no attribute 'BagLearner'
"""
# """My original submission:
import numpy as np, BagLearner as bl, LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False): self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, bags=20, kwargs={}, boost=False, verbose=False) for _ in range(0, 20)]
    def author(self): return "dstrube3"
    def add_evidence(self, data_x, data_y): [learner.add_evidence(data_x, data_y) for learner in self.learners]
    def query(self, points): return np.mean(np.array([learner.query(points) for learner in self.learners]), axis=0)
# """
