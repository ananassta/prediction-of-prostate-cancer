def t_to_int(t):
    res = []
    for i in range(t.size):
        if t[i] == '1':
            res.append(1)
        if t[i] == '2':
            res.append(2)
        if t[i] == '2b':
            res.append(3)
        if t[i] == '2c':
            res.append(4)
        if t[i] == '3':
            res.append(5)
        if t[i] == '3a':
            res.append(6)
        if t[i] == '3b':
            res.append(7)
        if t[i] == '4':
            res.append(8)
    return res

def nm_to_int(n):
    res = []
    for i in range(len(n)):
        res.append(int(n[i]))
    return res

def g_to_it(g):
    res = []
    for i in range(g.size):
        res.append(int(round(g[i])))
    return res

def t_to_class(t):
    res = ''
    if t == 1:
        res = '1'
    if t == 2:
        res = '2'
    if t == 3:
        res = '2b'
    if t == 4:
        res = '2c'
    if t == 5:
        res = '3'
    if t == 6:
        res = '3a'
    if t == 7:
        res = '3b'
    if t == 8:
        res = '4'
    return res