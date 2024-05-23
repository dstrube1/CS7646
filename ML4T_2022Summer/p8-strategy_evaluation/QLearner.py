""""""
"""
Template for implementing QLearner  (c) 2015 Tucker Balch

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
# Using numpy's random instead of Python's:
# https://realpython.com/python-random/


class QLearner(object):
    # From P7
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available.
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        # Hope it's not *required* for state and action to have terrible internal variable names s & a, respectively
        self.state = 0
        self.action = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.experience_tuple = []
        self.q_table = np.zeros([self.num_states, self.num_actions])

    def author(self):
        return 'dstrube3'

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """

        # Wouldn't have been my first choice for name of the function or parameter, but whatever

        self.state = s
        self.action = np.argmax(self.q_table[self.state])
        if self.verbose:
            print(f"query_set_stating: state = {self.state}, action = {self.action}")
        return self.action

    def query(self, s_prime, r):
        """
        Update the Q table and return an action

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """

        if self.verbose:
            print(f"querying: state' = {s_prime}, reward = {r}")

        # update the Q table
        self.q_table[self.state, self.action] = (1 - self.alpha) * self.q_table[self.state, self.action] + self.alpha \
                * (r + self.gamma * self.q_table[s_prime, np.argmax(self.q_table[s_prime])])

        # update experience tuple
        self.experience_tuple.append((self.state, self.action, s_prime, r))

        # Handle if dyna
        if self.dyna > 0:
            # Someone's in the kitchen with dyna. Someone's in the kitchen I kno-o-o-o-ow...
            experience_tuple_len = len(self.experience_tuple)
            if self.verbose:
                print(f"Handling dyna: experience_tuple_len' = {experience_tuple_len}")
            random_tuple = np.random.randint(experience_tuple_len, size=self.dyna)
            # TODO Is it possible to further vectorize?
            for i in range(0, self.dyna):
                temp_tuple = self.experience_tuple[random_tuple[i]]
                random_state = temp_tuple[0]
                random_action = temp_tuple[1]
                random_state_prime = temp_tuple[2]
                random_reward = temp_tuple[3]
                # update the Q table with randoms
                self.q_table[random_state, random_action] = (1 - self.alpha) * self.q_table[random_state,
                    random_action] + self.alpha * (random_reward + self.gamma * self.q_table[random_state_prime,
                    np.argmax(self.q_table[random_state_prime])])

        # Set to the new action (depending on rar)
        prob = np.random.uniform(0.0, 1.0)
        if prob < self.rar:
            self.action = np.random.random_integers(0, self.num_actions - 1)
        else:
            self.action = np.argmax(self.q_table[s_prime])

        if self.verbose:
            print(f"action = {self.action}")

        # Set to the new rar (based on decay)
        self.rar *= self.radr

        # Set to the new state
        self.state = s_prime

        return self.action
