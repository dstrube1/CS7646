"""Assess a betting strategy.  
  
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
import matplotlib.pyplot as plt
import numpy as np


def author():
	"""
	:return: The GT username of the student
	:rtype: str
	"""
	return "dstrube3"  # replace tb34 with your Georgia Tech username.


def gtid():
	"""
	:return: The GT ID of the student
	:rtype: int
	"""
	return 901081581  # replace with your GT ID number


def get_spin_result():
	"""
	Given a win probability between 0 and 1, the function returns whether the probability will result in a win.
	:return: The result of the spin.
	:rtype: bool
	"""
	result = False
	rando = np.random.random()
	# print("rando: ", rando)
	if rando <= win_prob:
		result = True
	return result


def test_code():
	"""
	Method to test your code
	"""
	# This wasn't very helpful:
	# https://en.wikipedia.org/wiki/Roulette#Bet_odds_table
	# Black...Odds against winning (American) =	 1+1⁄9 to 1 ?

	# This was a little helpful but not educational
	# https://www.roulettesites.org/rules/odds/
	# Roulette Bet - American Roulette Odds
	# Black = 47.4%
	# Outside Roulette Bet - American Roulette Odds
	# Black = 47.37%

	# More helpful:
	# https://www.roulettephysics.com/odds-of-red-or-black-in-a-row/
	# "On the European wheel, there are 37 numbers. 18 are black, and 18 are red. So the odds of red spinning are
	# 18/37 = 0.4865."
	# On an American wheel, there are 38 numbers; 0 and 00 are green, neither black nor red.
	# So, 18 out of 38 are black, which is close to 47.37% but not quite.
	# print("win_prob: ", win_prob)
	# for _ in range(count):
	# 	won = get_spin_result()
	# 	if won:
	# 		print("+", end='')  # test the roulette spin
	# 	else:
	# 		print("-", end='')
	# add your code here to implement the experiments
	# print()

	x_min = 0
	x_max = 300
	y_min = -256
	y_max = 100

	# Experiment 1, figure 1:
	for x in range(10):
		result = episode()
		plt.plot(result, label='Episode ' + str(x+1))
	plt.title('Figure 1: Winnings with unlimited bankroll over 10 episodes')
	plt.axis((x_min, x_max, y_min, y_max))
	plt.xlabel('Number of spins')
	plt.ylabel('Winnings')
	plt.legend(loc='lower right')
	plt.savefig('images/E1F1.png')
	plt.clf()

	# Experiment 1, figure 2
	winnings = np.zeros((1000, 1000))
	for x in range(1000):
		winnings[x] = episode()
	mean = np.mean(winnings, axis=0)
	std = np.std(winnings, axis=0)
	sum = mean + std
	diff = mean - std
	plt.title('Figure 2: Mean of winnings with unlimited bankroll over 1000 episodes')
	plt.axis((x_min, x_max, y_min, y_max))
	plt.xlabel('Number of spins')
	plt.ylabel('Winnings')
	plt.plot(mean, label='Mean of winnings')
	plt.plot(sum, label='Mean + standard deviation')
	plt.plot(diff, label='Mean - standard deviation')
	plt.legend(loc='lower right')
	plt.savefig('images/E1F2.png')
	plt.clf()

	# Experiment 1, figure 3
	winnings = np.zeros((1000, 1000))
	for x in range(1000):
		winnings[x] = episode()
	median = np.median(winnings, axis=0)
	std = np.std(winnings, axis=0)
	sum = median + std
	diff = median - std
	plt.title('Figure 3: Median of winnings with unlimited bankroll over 1000 episodes')
	plt.axis((x_min, x_max, y_min, y_max))
	plt.xlabel('Number of spins')
	plt.ylabel('Winnings')
	plt.plot(mean, label='Median of winnings')
	plt.plot(sum, label='Median + standard deviation')
	plt.plot(diff, label='Median - standard deviation')
	plt.legend(loc='lower right')
	plt.savefig('images/E1F3.png')
	plt.clf()

	# Experiment 2, figure 4
	winnings = np.zeros((1000, 1000))
	for x in range(1000):
		winnings[x] = episode(256)
	mean = np.mean(winnings, axis=0)
	# To find out the probability of winning (or not)
	# print("Experiment 2: mean: ", mean)
	std = np.std(winnings, axis=0)
	sum = mean + std
	diff = mean - std
	plt.title('Figure 4: Mean of winnings with limited bankroll over 1000 episodes')
	plt.axis((x_min, x_max, y_min, y_max))
	plt.xlabel('Number of spins')
	plt.ylabel('Winnings')
	plt.plot(mean, label='Mean of winnings')
	plt.plot(sum, label='Mean + standard deviation')
	plt.plot(diff, label='Mean - standard deviation')
	plt.legend(loc='lower right')
	plt.savefig('images/E2F4.png')
	plt.clf()

	# Experiment 2, figure 5
	winnings = np.zeros((1000, 1000))
	for x in range(1000):
		winnings[x] = episode(256)
	median = np.median(winnings, axis=0)
	std = np.std(winnings, axis=0)
	sum = median + std
	diff = median - std
	plt.title('Figure 5: Median of winnings with limited bankroll over 1000 episodes')
	plt.axis((x_min, x_max, y_min, y_max))
	plt.xlabel('Number of spins')
	plt.ylabel('Winnings')
	plt.plot(mean, label='Median of winnings')
	plt.plot(sum, label='Median + standard deviation')
	plt.plot(diff, label='Median - standard deviation')
	plt.legend(loc='lower right')
	plt.savefig('images/E2F5.png')
	plt.clf()


def episode(bankroll_limit=None):
	episode_winnings = 0  # 0$
	winning_limit = 80  # 80$
	episode_losings = 0
	bet_count = 0
	episode_limit = 1000
	# Must declare outside the while loop if we're going to print this later on at the end
	# bet_amount = 0
	winning_record = np.full(episode_limit, winning_limit)
	while episode_winnings < winning_limit and bet_count < episode_limit:  # 80$
		won = False
		bet_amount = 1  # 1$
		while not won:
			if bet_count >= episode_limit:
				return winning_record
			# wager bet_amount on black
			won = get_spin_result()
			if won:
				episode_winnings += bet_amount
			else:
				episode_winnings -= bet_amount
				episode_losings += bet_amount
				bet_amount *= 2
			winning_record[bet_count] = episode_winnings
			bet_count += 1

			if bankroll_limit is not None:
				if episode_winnings == -bankroll_limit:
					winning_record[bet_count:] = episode_winnings
					return winning_record
				if episode_winnings - bet_amount < -bankroll_limit:
					bet_amount = bankroll_limit + episode_winnings
	# print("episode_winnings: ", episode_winnings, end='')
	# print("; episode_losings: ", episode_winnings)  # , end='')
	# print("; bet_count: ", bet_count, end='')
	# print("; bet_amount: ", bet_amount)
	# if episode_winnings != 80:
	# 	print("episode_winnings != 80")
	# else:
	# 	print(".", end='')
	return winning_record  # bet_count < episode_limit


if __name__ == "__main__":
	win_prob = 18 / 38
	np.random.seed(gtid())  # do this only once
	test_code()
	"""count = 1_000  # _000
	for x in range(count):
		if x % 1000 == 0:
			print("-", end='')
		episode()"""
