import math
import time
import numpy as np

import DTLearner as dt
import LinRegLearner as lrl
from gen_data import best_4_dt, best_4_lin_reg


# compare two learners' rmse out of sample
def compare_os_rmse(learner1, learner2, x, y):
    """
    Compares the out-of-sample root mean squared error of your LinRegLearner and DTLearner.

    :param learner1: An instance of LinRegLearner
    :type learner1: class:'LinRegLearner.LinRegLearner'
    :param learner2: An instance of DTLearner
    :type learner2: class:'DTLearner.DTLearner'
    :param x: X data generated from either gen_data.best_4_dt or gen_data.best_4_lin_reg
    :type x: numpy.ndarray
    :param y: Y data generated from either gen_data.best_4_dt or gen_data.best_4_lin_reg
    :type y: numpy.ndarray
    :return: The root mean squared error of each learner
    :rtype: tuple
    """
    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6 * x.shape[0]))
    test_rows = x.shape[0] - train_rows

    # separate out training and testing data
    train = np.random.choice(x.shape[0], size=train_rows, replace=False)
    test = np.setdiff1d(np.array(range(x.shape[0])), train)
    train_x = x[train, :]
    train_y = y[train]
    test_x = x[test, :]
    test_y = y[test]

    # train the learners
    learner1.add_evidence(train_x, train_y)  # train it
    learner2.add_evidence(train_x, train_y)  # train it

    # evaluate learner1 out of sample
    pred_y = learner1.query(test_x)  # get the predictions
    rmse1 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    # evaluate learner2 out of sample
    pred_y = learner2.query(test_x)  # get the predictions
    rmse2 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

    return rmse1, rmse2


def test_code():
    """
    Performs a test of your code and prints the results
    """
    total_start = time.time()
    count = 0
    lr_fail_count = 0
    dt_fail_count = 0
    inner_min = 100
    inner_max = 0
    for i in range(10):
        for j in range(10):
            inner_start = time.time()
            lr_success, dt_success = sub_test_code(leafs=i, seed=j)
            if not lr_success:
                lr_fail_count += 1
            if not dt_success:
                dt_fail_count += 1

            inner_end = time.time()
            inner_seconds = inner_end - inner_start
            if inner_seconds >= 5:
                print("Each run should take less than 5 seconds:", inner_seconds)
            count += 1
            if inner_seconds < inner_min:
                inner_min = inner_seconds
            if inner_seconds > inner_max:
                inner_max = inner_seconds
    total_end = time.time()
    total_seconds = total_end - total_start
    print("test_code completed in {0:.3f} seconds".format(total_seconds))
    print("min: {0:.3f}".format(inner_min))
    print("max: {0:.3f}".format(inner_max))
    print("avg: {0:.3f}".format(total_seconds / count))
    print("lr_fail_count:", lr_fail_count, "; dt_fail_count:", dt_fail_count, "; count: ", count)

    # 100 x 100 =~ 360 s = 6 minutes
    # 50 x 200 =~ 600 s = 10 min; all with seed=0 failed, probably related to this from best_4_lin_reg
    #   y = seed * x[:, 0]
    #   (which has now been fixed):
    # 50 x 100 =~ 300 s = 5 min; failed at seed = 15 & 16, between leafs=14-49; reproducible


def sub_test_code(leafs=1, seed=1):
    lr_result = True
    dt_result = True
    # create two learners and get data
    lrlearner = lrl.LinRegLearner(verbose=False)
    dtlearner = dt.DTLearner(verbose=False, leaf_size=leafs)
    x, y = best_4_lin_reg(seed=seed)

    # compare the two learners
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)

    # share results
    # print()
    # print("best_4_lin_reg() results", end='')
    # print(f"RMSE LR    : {rmse_lr}")
    # print(f"RMSE DT    : {rmse_dt}")
    if rmse_lr < 0.9 * rmse_dt:
        # print("LR < 0.9 DT:  pass")
        # print("-", end='')
        pass
    else:
        print()
        print("best_4_lin_reg() results", end='')
        print("LR >= 0.9 DT:  fail; leafs=", leafs, "seed=", seed)
        lr_result = False
    # print("\n")

    # get data that is best for a random tree
    lrlearner = lrl.LinRegLearner(verbose=False)
    dtlearner = dt.DTLearner(verbose=False, leaf_size=leafs)
    x, y = best_4_dt(seed=seed)

    # compare the two learners
    rmse_lr, rmse_dt = compare_os_rmse(lrlearner, dtlearner, x, y)

    # share results
    # print()
    # print("best_4_dt() results", end='')
    # print(f"RMSE LR    : {rmse_lr}")
    # print(f"RMSE DT    : {rmse_dt}")
    if rmse_dt < 0.9 * rmse_lr:
        # print("DT < 0.9 LR:  pass")
        # print(".", end='')
        pass
    else:
        #print("best_4_dt() results", end='')
        #print("DT >= 0.9 LR:  fail; leafs=", leafs, "seed=", seed)
        #print(".", end='')
        dt_result = False
    # print("\n")
    return lr_result, dt_result


if __name__ == "__main__":
    test_code()
