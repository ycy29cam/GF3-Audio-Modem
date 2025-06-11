import numpy as np
from pathlib import Path
from typing import Tuple

FILE_BIN_DIR = "multi-file-test/examples.npy"
SUPPORTED_EXTENSIONS = {'.txt', '.wav', '.tiff', '.html', '.bmp'}


def load_file(path: str) -> Tuple[bytes, str]:
    """Return raw bytes and filename (without folders)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError('Unsupported extension')
    data = p.read_bytes()
    return data, p.name

def build_header(filename: str, nbytes: int) -> bytes:
    filename_b = filename.encode('utf-8') + b'\0'
    size_b = str(nbytes).encode('utf-8') + b'\0'
    return filename_b + size_b

def assemble_stream(path: str) -> bytes:
    data, name = load_file(path)
    header = build_header(name, len(data))
    return header + data

def bytes_to_uint8(byte_stream: bytes) -> np.ndarray:
    return np.frombuffer(byte_stream, dtype=np.uint8)

def uint8_to_bits(uint8_arr: np.ndarray) -> np.ndarray:
    """Return 1‑D array of bits (0/1), MSB first per byte."""
    return np.unpackbits(uint8_arr, bitorder='big')

def encode_file_to_bits(path: str) -> np.ndarray:
    stream = assemble_stream(path)
    u8 = bytes_to_uint8(stream)
    bits = uint8_to_bits(u8)
    return bits

file1 = encode_file_to_bits("multi-file-test/example.txt")
file2 = encode_file_to_bits("multi-file-test/example.tiff")
file3 = encode_file_to_bits("multi-file-test/example.bmp")
file4 = encode_file_to_bits("multi-file-test/example.html")
bin = np.concatenate((file1, file2, file3, file4))
np.save(FILE_BIN_DIR, np.hstack(np.array(bin)))