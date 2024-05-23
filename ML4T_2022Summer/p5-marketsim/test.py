# test.py

""" Test1:
lst = [i for i in range(0,20)]
print("lst: ", lst)

i = -1
for x in range(0, i):
    print("what happens? ", x) # Nothing
print("Can I print i?", i) # yes
print("How about this?", i, ", huh?") # yes, and it automatically adds a space

print("Done")
"""

""" Test2:
in /usr/local/lib/python3.9/site-packages/numpy/lib/function_base.py :
starting at line 2829, replace this:
    c /= stddev[:, None]
    c /= stddev[None, :]

with this:
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            c /= stddev[:, None]
            c /= stddev[None, :]
        except Warning as e:
            print('warning')
"""

""" Test3:
def fileOrString(x):
    print("File type: ", type(x))
    # if isinstance(x, type(str())):
    if isinstance(x, str):
        print("string")
    else:
        print("not string")

s = "s"
fileOrString(s)
f = open("test.py", "r")
fileOrString(f)
fileOrString(1)
fileOrString(None)
"""

""" Test4:
import os
dir_list = os.listdir("./orders/")
for in_file in dir_list:
    print(in_file)
"""

""" Test5:
"""
i = 0.49
for x in range(1000):
	i *= 0.49 # off by 1?


print("i = ", i)
