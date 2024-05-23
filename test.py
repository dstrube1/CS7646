import numpy as np
#python -m pip install --upgrade pip
#python -m pip install numpy

#import time
from time import time

def how_long(func, *args):
	"""Execute function with given arguments, and measure execution time."""
	t1 = time()
	result = func(*args) #all arguments are passed in as-is
	t2 = time()
	return result, t2 - t1
	
def manual_mean(arr):
	sum = 0
	for i in range(0, arr.shape[0]): #for each row
		for j in range(0, arr.shape[1]): #for each column
			sum += arr[i,j]
	return sum / arr.size

def numpy_mean(arr):
	"""Compute mean (average) using NumPy"""
	return arr.mean()
	
def test():
	"""
	# seed the random number generator
	np.random.seed(693)
	# 5x4 random integers in [0,10)
	# []: inclusive
	# (): exclusive
	a = np.random.randint(0,10, size=(5,4)) 
	print("2D Array:\n",a)
	print("Sum all elements: ", a.sum())
	
	# iterate over rows, compute sum of each column:
	print("Sum of each column:\n", a.sum(axis=0))
	
	# iterate over columns, compute sum of each row:
	print("Sum of each row:\n", a.sum(axis=1))
	
	#Statistics: min, max, mean (across rows, cols, and overall)
	print("Minimum of each column:\n", a.min(axis=0))
	print("Maximum of each row:\n", a.max(axis=1))
	print("Mean of all elements:\n", a.mean())
	
	#Index of max value:
	print("Index of max in a 2D array:\n", a.argmax())
	b = np.random.randint(0,10, size=(5,1)) 
	print("2D random array (5x1):\n",b)
	print("Index of max in a 2D random array: ", b.argmax())
	
	#32-bit integer array
	c = np.array([9,6,2,3,12,14,7,10], dtype=np.int32)
	print("1D non-random array: ",c)
	print("Max: ", c.max())
	print("Index of max in a 1D non-random array: ", c.argmax())
	"" "
	
	#Using just "import time"
	#t1 = time.time()
	#print("ML4T - testing time.time()")
	#t2 = time.time()
	#print("The time taken by the print statement: ", t2-t1, " seconds")
	
	# A large array:
	nd1 = np.random.random((1000, 1000))
	res_manual, t_manual = how_long(manual_mean, nd1)
	res_numpy, t_numpy = how_long(numpy_mean, nd1)
	#print("Manual: {:.6f} ({:.3f} secs) vs NumPy: {:.6f} ({:.3f} secs)".format(res_manual, t_manual, res_numpy, t_numpy))
	
	assert abs(res_manual - res_numpy) <= 10e-6, "Results aren't equal!"
	
	#Compute speedup
	speedup = t_manual / t_numpy
	#print("NumPy mean is ", speedup, " times faster than manual for loops.")
	
	a = np.random.randint(0,100, size=(5,4)) 
	print("Array: \n",a)
	#Accessing element at position (3,2)
	element = a[3,2]
	print("Element at position (3,2): ",element)
	
	#Elements in defined range
	print("Elements in defined range [0,1:3]: 0th row, items [1-3):\n", a[0,1:3])
	
	#Top-left corner
	print("Top-left corner:\n", a[0:2, 0:2])
	
	#Slicing
	#Note: Slice n:m:t specifies a range that starts at n, and stops before m, in steps of size t
	#t  = 2: every other column
	print("More slicing:\n", a[:, 0:3:2]) #will select columns 0, 2 for every row
	
	#Assigning a value to a particular location
	a[0,0] = 100
	print("Modified array - replaced on element: \n", a)
	
	#Assigning a value to an entire row 
	a[0,:] = 200
	print("Modified array - replaced row: \n", a)
	
	#Assigning a list of values to an entire column
	a[:,3] = [1,2,3,4,5]
	print("Modified array - replaced column from list: \n", a)
	
	#1D array of ints
	a = np.random.randint(0,100, size=(5))
	print("1D array of ints:",a)
	
	#accessing using list of indices
	indices = np.array([1,1,2,3])
	print("accessing using list of indices:", a[indices])
	"" "
	
	a = np.array([(20,25,10,23,26,32,10,5,0),(0,2,50,20,0,1,28,5,0)])
	print("array: ", a)
	mean = a.mean()
	print("mean: ", mean)
	
	#for each value in the array, compare it with the mean;
	#if the value is less than the mean, retain the value
	print("a[a < mean]: ", a[a<mean])
	#if the value is less than the mean, replace the value with the mean
	a[a<mean] = mean
	print("a[a < mean] = mean: ", a)
	"""
	
	a = np.array([(1,2,3,4,5),(10,20,30,40,50)])
	print("array a:\n", a)
	print("Multiply by 2:\n", 2 * a)
	print("Divide by 2:\n", a / 2)
	b = np.array([(100,200,300,400,500),(1,2,3,4,5)])
	print("array b:\n", b)
	#[maths: add, subtract, etc]
	#When [maths]ing two arrays, they should be the same size, else might get errors
	print("a + b:\n", a + b)
	print("a * b:\n", a * b) #not matrix multiplication, but element-by-element multiply
	print("a / b:\n", a / b) #not matrix multiplication, but element-by-element multiply
	
if __name__ == "__main__":
	test()
	print("Done")
	
"""

4T1BE46K08U786554
2008 Toyota Camry
197500 mi
LE

This 2008 Toyota Camry is in excellent condition. It comes with a CD/MP3/WMA disc player, 2 12V/120W charging ports, AM/FM1/FM2 radio presets, and an auxiliary audio input.

5YJ3E1EA7JF016057
2018 Tesla
77k
XGE702

880 Whitehawk Trail NW, Lawrenceville, GA 30043

"""