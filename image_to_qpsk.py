from PIL import Image
import io
#import pyldpc
import numpy as np
import matplotlib.pyplot as plt
from Constants import *
from emitter import chirp
from py.ldpc import code    # adjust the import path if needed



def image_to_bitstream(image_path):
    # Load image
    img = Image.open(image_path)
    # e.g. img = img.resize((128,128))      # optionally resize

    # Convert to raw bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    byte_data = buf.getvalue()

    # Flatten to bits
    bitstream = []
    for byte in byte_data:
        for b in range(8):
            bit = (byte >> (7 - b)) & 0x1
            bitstream.append(bit)

    return bitstream

"""
def ldpc_encode(bitstream, n_code=1024, d_v=3, d_c=15, systematic=True):
    
    #Encodes a bitstream with a regular (n_code, k) LDPC code.

    #- n_code: length of each coded word (n)
    #- d_v   : variable-node degree (columns of H)
    #- d_c   : check-node degree    (rows of H)
    #- systematic: if True, G will be systematic (data bits appear unchanged).
    
    # Build H and G (parity and generator matrix)
    H, G = pyldpc.make_ldpc(n_code, d_v, d_c,
                            systematic=systematic,
                            sparse=False)

    # Discover k from G’s shape
    _, k = G.shape

    # Split your bitstream into blocks of length k
    codewords = []
    for i in range(0, len(bitstream), k):
        u = bitstream[i : i + k]
        # Zero-pad the last block
        if len(u) < k:
            u = u + [0]*(k - len(u))
        # Return an array of length n_code
        cw = pyldpc.encode(G, np.array(u), snr=10)
        codewords.extend(cw.tolist())

    return codewords
"""

def ldpc_encode(bitstream,
                standard='802.16',   # or '802.11n'
                rate='1/2',
                z=27,                # expansion factor (e.g. 27 for 11n)
                proto='A'):
    """
    Encodes a bitstream using the protograph-based LDPC from our C-library + Python wrapper.

    - standard: '802.16' or '802.11n'
    - rate    : code rate string, e.g. '1/2', '2/3', etc.
    - z       : protograph expansion factor
    - proto   : protograph variant, usually 'A' or 'B'
    """
    # 1) instantiate the code
    mycode = code(standard, rate, z, proto)

    # 2) figure out how many data bits (K)
    H = mycode.pcmat()  # full parity-check matrix, shape (M, N)
    M, N = H.shape
    K = N - M  # true # of info bits

    # 3) slice into blocks of K and encode
    coded_bits = []
    for i in range(0, len(bitstream), K):
        block = bitstream[i:i+K]
        if len(block) < K:
            block = block + [0]*(K - len(block))
        codeword = mycode.encode(np.array(block))
        coded_bits.extend(codeword.tolist())

    return coded_bits


def bits_to_qpsk(coded_bits):
    symbols = []
    for i in range(0, len(coded_bits), 2):
        b0, b1 = coded_bits[i], coded_bits[i + 1] if i + 1 < len(coded_bits) else 0
        if (b0,b1) == (0,0): s =  1 + 1j
        elif (b0,b1) == (0,1): s = -1 + 1j
        elif (b0,b1) == (1,1): s = -1 - 1j
        else:                 s =  1 - 1j
        symbols.append(s/np.sqrt(2))
    return np.array(symbols)

# _______________________________________________________________________________________#

# Usage:

# PNG to bitstream
bitstream = image_to_bitstream('test_image.png')  # Replace with PNG path

# LDPC encoding
"""
try:
    # (1024, k) LDPC code with dv=3, dc=15
    coded_bits = ldpc_encode(bitstream,
                             n_code=1024,
                             d_v=3,      # small column weight
                             d_c=16,     # needs to be a multiple of n_code
                             systematic=True)

    #coded_bits = ldpc_encode(bitstream)        # this also works

except ImportError:
    # fallback if pyldpc isn’t installed
    coded_bits = bitstream
"""

coded_bits = ldpc_encode(bitstream)

# Map bits to QPSK symbols
qpsk_symbols = bits_to_qpsk(coded_bits)



# Visualise plot
plt.figure(figsize=(6,6))
plt.scatter(qpsk_symbols.real, qpsk_symbols.imag, s=5)
plt.axhline(0, lw=1)
plt.axvline(0, lw=1)
plt.title('QPSK Constellation')
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.grid(True)
plt.gca().set_aspect('equal', 'box')
plt.show()
