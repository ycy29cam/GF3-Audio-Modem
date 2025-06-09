import numpy as np
from pathlib import Path
from typing import Tuple

FILE_DIR = "3000/example.txt"
FILE_BIN_DIR = "3000/example.npy"

def load_file(path: str) -> Tuple[bytes, str]:
    """Return raw bytes and filename (without folders)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() not in {'.txt', '.wav', '.tiff', '.tif'}:
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

bin = encode_file_to_bits(FILE_DIR)
np.save(FILE_BIN_DIR, bin)