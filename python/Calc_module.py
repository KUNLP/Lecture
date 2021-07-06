class Calc:
    
    my_name = "I'm Calc Module!"
    
    def add(self, a, b):
        result = a + b
        print("{0:f} + {1:f} = {2:f}".format(a,b,result))
    def sub(self, a, b):
        result = a - b
        print("{0:f} - {1:f} = {2:f}".format(a,b,result))
    def mul(self, a, b):
        result = a * b
        print("{0:f} * {1:f} = {2:f}".format(a,b,result))
    def div(self, a, b):
        if b != 0:
            result = a / b
            print("{0:f} / {1:f} = {2:f}".format(a,b,result))
        else:
            print("Divided by zero!")




