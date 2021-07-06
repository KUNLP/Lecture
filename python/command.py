import sys

num = 1

for arg in sys.argv:
    print("argument #"+str(num)+": "+arg)
    num+=1

