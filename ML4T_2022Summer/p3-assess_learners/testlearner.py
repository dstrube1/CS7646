""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
"""

import math
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il

# Contains the code necessary to run experiments and perform additional independent testing


def clean_istanbul():
    """"# Hard way
    line_count = -1
    read_list = list()
    for s in inf.readlines():
        line_count += 1
        if line_count == 0:
            continue
        read_list.append(s.strip().split(",")[1:])
    data = np.array([list(map(float, s)) for s in read_list])
    # """

    # Easy way, from grade_learners.py
    data = np.genfromtxt(inf, delimiter=',')
    data = data[1:, 1:]
    return data


def training(training_learner):
    training_learner.add_evidence(train_x, train_y)  # train it
    print("author: ", training_learner.author())

    # """
    # evaluate in sample
    pred_y = training_learner.query(train_x)  # get the predictions
    # Root-mean-square error
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results: ", end='')
    print(f"RMSE: {rmse}")
    # print(f"RMSE: %.2e" % rmse, end='')
    # Better-looking and more intuitive float formatting:
    # https://appdividend.com/2021/03/31/how-to-format-float-values-in-python/
    print("RMSE: {:.2e}".format(rmse))

    # Pearson product-moment correlation coefficients
    # https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    correlation_coefficients = np.corrcoef(pred_y, y=train_y)
    print(f"; corr: {correlation_coefficients[0, 1]}")
    # """

    # """
    # evaluate out of sample
    pred_y = training_learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    print("Out of sample results: ", end='')
    # print(f"RMSE: {rmse}")
    print(f"RMSE: %.2e" % rmse, end='')
    correlation_coefficients = np.corrcoef(pred_y, y=test_y)
    print(f"; corr: {correlation_coefficients[0, 1]}")
    # """


def test_drives():
    # create a learner and train it
    # LinRegLearner
    print("\nLinRegLearner")
    learner = lrl.LinRegLearner(verbose=True)
    training(learner)

    # DTLearner
    # print("\nDTLearner")
    learner = dt.DTLearner(leaf_size=train_y.shape[0], verbose=False)
    # training(learner)

    # learner = dt.DTLearner(leaf_size=train_y.shape[0], verbose=True)  # constructor
    learner.add_evidence(train_x, train_y)  # training step
    Y_pred = learner.query(test_x)
    # DTLearner tests must complete in 10 seconds each (e.g., 50 seconds for 5 tests)

    # RTLearner
    print("\nRTLearner")
    learner = rt.RTLearner(leaf_size=1000, verbose=True)
    training(learner)

    learner = rt.RTLearner(leaf_size=1000, verbose=True)  # constructor
    learner.add_evidence(train_x, train_y)  # training step
    Y_pred = learner.query(test_x)
    # RTLearner tests must complete in 3 seconds each (e.g., 15 seconds for 5 tests

    # BagLearner
    print("\nBagLearner")
    learner = bl.BagLearner(verbose=True)
    training(learner)

    # learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1000}, bags=20, boost=False,
    #                            verbose=False)
    learner = bl.BagLearner(learner=il.InsaneLearner, kwargs={}, bags=1000, boost=False,
                            verbose=True)
    learner.add_evidence(train_x, train_y)
    Y = learner.query(test_x)
    # BagLearner tests must complete in 10 seconds each (e.g., 100 seconds for 10 tests)

    # InsaneLearner
    print("\nInsaneLearner")
    learner = il.InsaneLearner(verbose=True)
    training(learner)

    learner = il.InsaneLearner(verbose=True)  # constructor
    learner.add_evidence(train_x, train_y)  # training step
    Y = learner.query(test_x)
    print(learner.author())
    # InsaneLearner must complete in 10 seconds each


def ex1(ex1_train_x, ex1_train_y, ex1_test_x, ex1_test_y):
    # Experiment 1
    rmse_training = []
    rmse_testing = []
    total_start = time.time()
    for i in range(20):
        start = time.time()
        # Make a learner
        learner = dt.DTLearner(leaf_size=i)  # , verbose=True)
        # Train the learner
        learner.add_evidence(ex1_train_x, ex1_train_y)
        # Predict based on training
        predicted_training = learner.query(train_x)
        # Root-mean-square error of training prediction
        rmse_training.append(math.sqrt(((ex1_train_y - predicted_training) ** 2).sum() / ex1_train_y.shape[0]))
        # Predict based on testing
        predicted_testing = learner.query(ex1_test_x)
        end = time.time()
        seconds = end - start
        if seconds > 10:
            print("Experiment 1 learner", i, "completed in {0:.3f} seconds".format(seconds))
            # No learner takes more than 1 second; any point graphing the times?
        # Root-mean-square error of testing prediction
        rmse_testing.append(math.sqrt(((ex1_test_y - predicted_testing) ** 2).sum() / ex1_test_y.shape[0]))
    total_end = time.time()
    total_seconds = total_end - total_start

    plt.figure(1)
    plt.plot(rmse_training)
    plt.plot(rmse_testing)
    plt.title('Experiment 1: Decision Tree, RMSE vs Leaf Size')
    plt.xlabel('Leaf Sizes')
    plt.ylabel('RMSEs')
    plt.legend(labels=['In Sample RMSE', 'Out of Sample RMSE'], loc='best')
    plt.grid(visible=True)
    # plt.show()
    plt.savefig('images/Figure1')
    plt.clf()

    print("Experiment 1 completed in {0:.3f} seconds".format(total_seconds))


def ex2(ex2_train_x, ex2_train_y, ex2_test_x, ex2_test_y):
    # Experiment 2
    rmse_training = []
    rmse_testing = []
    testing_times = []
    total_start = time.time()
    for i in range(20):
        # Make a learner
        learner_start = time.time()
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size': i}, bags=10)

        # Train the learner
        learner.add_evidence(ex2_train_x, ex2_train_y)

        # Predict based on training
        predicted_training = learner.query(train_x)

        # Root-mean-square error of training prediction
        rmse_training.append(math.sqrt(((ex2_train_y - predicted_training) ** 2).sum() / ex2_train_y.shape[0]))

        # Predict based on testing
        testing_start = time.time()
        predicted_testing = learner.query(ex2_test_x)
        testing_end = time.time()
        learner_end = time.time()
        testing_seconds = testing_end - testing_start
        testing_times.append(testing_seconds)
        # learner_end is practically equal to testing_end, but it helps to keep them in separate variables
        learner_seconds = learner_end - learner_start
        if learner_seconds > 10:
            print("Experiment 2 learner", i, "completed in {0:.3f} seconds".format(learner_seconds))
        # Root-mean-square error of testing prediction
        rmse_testing.append(math.sqrt(((ex2_test_y - predicted_testing) ** 2).sum() / ex2_test_y.shape[0]))
    total_end = time.time()
    total_seconds = total_end - total_start

    plt.figure(20)
    plt.plot(rmse_training)
    plt.plot(rmse_testing)
    plt.title('Experiment 2: Bag of Decision Tree Learners, RMSE vs Leaf Size')
    plt.xlabel('Leaf Sizes')
    plt.ylabel('RMSEs')
    plt.legend(labels=['In Sample RMSE', 'Out of Sample RMSE'], loc='best')
    plt.grid(visible=True)
    # plt.show()
    plt.savefig('images/Figure2a')
    plt.clf()

    plt.figure(21)
    plt.plot(testing_times)
    plt.title('Experiment 2: Bag of Decision Tree Learners, Testing Time vs Leaf Size')
    plt.xlabel('Leaf Sizes')
    plt.ylabel('Testing Times')
    plt.legend(labels=['In Sample RMSE', 'Out of Sample RMSE'], loc='best')
    plt.grid(visible=True)
    # plt.show()
    plt.savefig('images/Figure2b')
    plt.clf()

    print("Experiment 2 completed in {0:.3f} seconds".format(total_seconds))


def ex3(ex3_train_x, ex3_train_y, ex3_test_x, ex3_test_y):
    # Experiment 3
    dt_maes = []
    rt_maes = []
    dt_training_times = []
    rt_training_times = []
    total_start = time.time()
    for i in range(0, 200):
        # Make a learner
        learner_start = time.time()
        learner = dt.DTLearner(leaf_size=i)

        # Train the learner
        training_start = time.time()
        learner.add_evidence(ex3_train_x, ex3_train_y)
        training_end = time.time()
        training_seconds = training_end - training_start
        dt_training_times.append(training_seconds)

        # Predict based on testing
        predicted_testing = learner.query(ex3_test_x)

        learner_end = time.time()
        learner_seconds = learner_end - learner_start
        if learner_seconds > 10:
            print("Experiment 3 DT learner", i, "completed in {0:.3f} seconds".format(learner_seconds))

        # Mean Absolute Error
        mae = (abs(ex3_test_y - predicted_testing)).sum() / ex3_test_y.shape[0]
        dt_maes.append(mae)
    for j in range(0, 200):
        # Make a learner
        learner_start = time.time()
        learner = rt.RTLearner(leaf_size=j)

        # Train the learner
        training_start = time.time()
        learner.add_evidence(ex3_train_x, train_y)
        training_end = time.time()
        training_seconds = training_end - training_start
        rt_training_times.append(training_seconds)

        # Predict based on testing
        predicted_testing = learner.query(ex3_test_x)

        learner_end = time.time()
        learner_seconds = learner_end - learner_start
        if learner_seconds > 10:
            print("Experiment 3 RT learner", j, "completed in {0:.3f} seconds".format(learner_seconds))

        # Mean Absolute Error
        mae = (abs(ex3_test_y - predicted_testing)).sum() / ex3_test_y.shape[0]
        rt_maes.append(mae)

    dt_maes_0 = []
    rt_maes_0 = []
    for h in range(1, 21):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={'leaf_size': 9}, bags=h)

        # Train the learner
        learner.add_evidence(ex3_train_x, ex3_train_y)

        # Predict based on testing
        predicted_testing = learner.query(ex3_test_x)

        # Mean Absolute Error
        mae = (abs(ex3_test_y - predicted_testing)).sum() / ex3_test_y.shape[0]

        dt_maes_0.append(mae)

    for k in range(1, 21):
        learner = bl.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size': 9}, bags=k)

        # Train the learner
        learner.add_evidence(ex3_train_x, ex3_train_y)

        # Predict based on testing
        predicted_testing = learner.query(ex3_test_x)

        # Mean Absolute Error
        mae = (abs(ex3_test_y - predicted_testing)).sum() / ex3_test_y.shape[0]

        rt_maes_0.append(mae)

    total_end = time.time()
    total_seconds = total_end - total_start

    plt.figure(30)
    plt.plot(dt_maes)
    plt.plot(rt_maes)
    plt.title('Experiment 3: Mean Absolute Error, DecisionTree vs RandomTree')
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error')
    plt.legend(labels=['DT MAE', 'RT MAE'], loc='best')
    plt.grid(visible=True)
    # plt.show()
    plt.savefig('images/Figure3a')
    plt.clf()

    plt.figure(31)
    plt.plot(dt_training_times)
    plt.plot(rt_training_times)
    plt.title('Experiment 3: Training Time, DecisionTree vs RandomTree')
    plt.xlabel('Leaf Size')
    plt.ylabel('Training Times')
    plt.legend(labels=['DT', 'RT'], loc='best')
    plt.grid(visible=True)
    # plt.show()
    plt.savefig('images/Figure3b')
    plt.clf()

    plt.figure(32)
    plt.plot(dt_maes_0)
    plt.plot(rt_maes_0)
    plt.title('Experiment 3: MAE, DecisionTree vs RandomTree, via BagLearner')
    plt.xlabel('Bags')
    plt.ylabel('Mean Absolute Error')
    plt.legend(labels=['DT MAE', 'RT MAE'], loc='best')
    plt.grid(visible=True)
    # plt.show()
    plt.savefig('images/Figure3c')
    plt.clf()

    print("Experiment 3 completed in {0:.3f} seconds".format(total_seconds))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])

    if sys.argv[1] == 'Data/Istanbul.csv':
        data = clean_istanbul()
    else:
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    # If we want to run through with all training no testing:
    # train_rows = data.shape[0]
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"test_x.shape: {train_x.shape}")
    print(f"test_y.shape: {train_y.shape}")

    print(f"test_x.shape: {test_x.shape}")
    print(f"test_y.shape: {test_y.shape}")

    # test_drives()

    # Experiment 1
    ex1(train_x, train_y, test_x, test_y)

    # Experiment 2
    # ex2(train_x, train_y, test_x, test_y)

    # Experiment 3
    # ex3(train_x, train_y, test_x, test_y)
