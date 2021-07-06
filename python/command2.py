import sys

str_list = []

for arg in sys.argv[1:]:
    str_list.append(arg.upper())
    
print(" ".join(str_list))
