import ldpc_jossy.py.ldpc as ldpc
import numpy as np
c = ldpc.code()
x = c.encode(np.random.randint(0, 2, c.K))
y = 10 * (.5 - x)
app, it = c.decode(y)
print(it)  # -> 0

#read up on how this works tonight, very happy it works and i've downloaded it