""""""
import math

"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

Student Name: David Strube
GT User ID: dstrube3
GT ID: 901081581
"""

import numpy as np


# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best_4_lin_reg(seed=1489683273):
    """
    Returns data that performs significantly better with LinRegLearner than DTLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    :param seed: The random seed for your data generation.
    :type seed: int
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.
    :rtype: numpy.ndarray
    """
    if seed is None:
        # https://numpy.org/doc/stable/reference/constants.html
        seed = np.pi
    np.random.seed(seed)
    row_count = np.random.randint(10, 1001)
    column_count = np.random.randint(2, 11)
    x = np.random.random(size=(row_count, column_count))

    y = np.e * x[:, 0]
    return x, y


# this function should return a dataset (X and Y) that will work
# better for decision trees than linear regression
def best_4_dt(seed=1489683273):
    """
    Returns data that performs significantly better with DTLearner than LinRegLearner.
    The data set should include from 2 to 10 columns in X, and one column in Y.
    The data should contain from 10 (minimum) to 1000 (maximum) rows.

    :param seed: The random seed for your data generation.
    :type seed: int
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.
    :rtype: numpy.ndarray
    """
    if seed is None:
        seed = np.pi
    np.random.seed(seed)

    # Maximum of column count, minimum of row count:
    no_more_no_less = 10

    row_count = no_more_no_less
    column_count = no_more_no_less
    # 46% failure rate
    # x = np.random.random(size=(row_count, column_count)) * math.pow(np.pi, np.e)
    # 42%
    # x = np.random.random(size=(row_count, column_count)) * 2 - 1
    # 38%
    # x = np.random.random(size=(row_count, column_count)) * 3 - 1
    # 18%
    # x = np.random.random(size=(row_count, column_count)) * 300 - 100
    # 14%
    # x = np.random.random(size=(row_count, column_count)) * 50 - 25
    # 10%
    # x = np.random.random(size=(row_count, column_count)) * 200 - 100
    # 10%
    # x = np.random.random(size=(row_count, column_count)) * 2000 - 1000
    # 10%
    # x = np.random.random(size=(row_count, column_count)) * 1000 - 500
    # 10%
    # x = np.random.random(size=(row_count, column_count)) * 10000 - 5000
    # total fail:
    # x = np.random.random(size=(row_count * 2, column_count * 2)) * 1000 - 500
    # total fail:
    # x = np.random.random(size=(row_count / 2, column_count / 2)) * 1000 - 500
    # 10%:
    x = np.random.random(size=(row_count, column_count)) * 100 - 50

    # 100%:
    # y = np.sin(0) * x[:, 0]
    # 82%:
    """
    y = x[:, 0]
    i = True
    count = 0
    for temp in x[:, 0]:
        if i:
            i = False
        else:
            y[count] = -temp
            i = True
        count += 1
    """
    # 58%:
    # y = np.sin(np.pi / 2) * x[:, 0]
    # 44%:
    """
    y = np.random.random(size=row_count)  
    col = 0
    for row in range(row_count):
        # 44%:
        y[row] = x[row, col]
        # 80%:
        # y[row] = x[row, row % 2]
        if col < column_count:
            col += 1
        else:
            col = 0
    """

    # 10%:
    # From Numpy documentation: "Evenly spaced values within a given interval"
    #y = np.arange(no_more_no_less)

    # 0% failure rate in local tests:
    y = x[:, 0]
    count = 0
    for temp in x[:, 0]:
        if temp < 0:
            y[count] = math.sqrt(-temp)
        else:
            y[count] = math.sqrt(temp)
        count += 1

    return x, y


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "dstrube3"


if __name__ == "__main__":
    # print("they call me Tim.")
    pass