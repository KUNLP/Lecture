#-*- coding: utf-8 -*-

import sys

arg_num = len(sys.argv)
if arg_num !=3 :
    print("Usage: python cp.py file1 file2")
    sys.exit()

in_name = sys.argv[1]
out_name = sys.argv[2]

in_f=open(in_name,"rb")    
out_f=open(out_name,"wb")
    
data=in_f.read()
out_f.write(data)

in_f.close()
out_f.close()
