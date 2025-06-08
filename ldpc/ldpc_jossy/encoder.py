import ldpc_jossy.py.ldpc as ldpc
import numpy as np

LDPC_STD = "802.11n"
LDPC_RATE = "1/2"
LDPC_Z = 81

c = ldpc.code(LDPC_STD, LDPC_RATE, LDPC_Z)
x = c.encode(np.random.randint(0, 2, c.K))
print(x)
print(len(x))
y = 10 * (.5 - x)
print(y) 

app, it = c.decode(y)
print(len(app))
print(it)

