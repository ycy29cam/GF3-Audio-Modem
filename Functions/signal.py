import numpy as np

def signal_constructor(segments: list, block_length, cyclic_prefix_length=0, prefix_indices=[]):
    if not isinstance(segments, list) or not all(isinstance(seg, np.ndarray) for seg in segments):
        raise TypeError("All segments must be a list of 1D numpy arrays.")

    if not all(seg.ndim == 1 for seg in segments):
        raise ValueError("Each segment must be a 1D array.")

    if not isinstance(block_length, int) or block_length <= 0:
        raise ValueError("block_length must be a positive integer.")

    if not isinstance(cyclic_prefix_length, int) or cyclic_prefix_length < 0:
        raise ValueError("cyclic_prefix_length must be a non-negative integer.")

    if not all(isinstance(i, int) and 0 <= i < len(segments) for i in prefix_indices):
        raise IndexError("prefix_indices must contain valid indices of segments.")

    full_signal = []

    for i, seg in enumerate(segments):
        if i in prefix_indices:
            if len(seg) != block_length:
                raise ValueError(f"Segment at index {i} has length {len(seg)}, expected {block_length} for cyclic prefixing.")
            if cyclic_prefix_length > 0:
                prefix = seg[-cyclic_prefix_length:]
                print(f"Inserting cyclic prefix at segment {i}: prefix length = {cyclic_prefix_length}")
                seg = np.concatenate((prefix, seg))
        full_signal.append(seg)

    final_signal = np.concatenate(full_signal)
    print(f"Signal constructed successfully. Total length: {len(final_signal)} samples.")
    return final_signal
