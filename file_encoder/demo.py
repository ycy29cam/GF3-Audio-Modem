import numpy as np
from pathlib import Path
from typing import Tuple
from file_encoder import encode_file_to_bits

SAVE_NAME = "example"

bits = encode_file_to_bits('file_encoder/files/example.txt')
print(f'Encoded {bits.size} bits, first 64 bits:\n', bits[:64])

byte_array = np.packbits(bits)

file_name_terminate = np.where(byte_array == 0)[0][0]
file_name = byte_array[:file_name_terminate].tobytes().decode("utf-8")

file_size_terminate = np.where(byte_array == 0)[0][1]
file_size = int(byte_array[file_name_terminate + 1:file_size_terminate].tobytes().decode("utf-8"))

file_content = byte_array[(file_size_terminate + 1):(file_size_terminate + int(file_size) + 1)]

if ".tiff" in file_name:
        with open(f"{SAVE_NAME}.tiff", "wb") as f:
            f.write(file_content.tobytes())
elif ".wav" in file_name:
    with open(f"{SAVE_NAME}.wav", "wb") as f:
        f.write(file_content.tobytes())
elif ".txt" in file_name:
     with open(f"{SAVE_NAME}.txt", "wb") as f:
          f.write(file_content.tobyte())
else:
    print("Error in file name extraction")