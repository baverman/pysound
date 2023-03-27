from asound import listen

def boo(etype, value):
    print(etype, value)


listen('16:0', boo)
