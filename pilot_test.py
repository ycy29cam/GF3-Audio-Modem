import numpy as np

def compare_pilot_symbols( num_blocks=50):
    """
    Compares the first N frequency blocks of pilot symbols from a .npy file and a .txt file.

    Args:
        npy_file (str): The path to the .npy file.
        txt_file (str): The path to the .txt file.
        num_blocks (int): The number of frequency blocks (rows) to compare.

    Returns:
        bool: True if the first N blocks are equal, False otherwise.
    """
    try:
        # Load the pilot symbols from the .npy file
        pilot_symbols_npy = np.load("pilot_symbols.npy")

        # Load the pilot symbols from the .txt file
        # This assumes the text file is delimited by whitespace.
        # If your delimiter is different (e.g., a comma), you can add the delimiter argument:
        # pilot_symbols_txt = np.loadtxt(txt_file, delimiter=',')
        pilot_symbols_2 = np.load("foo.npy")


        # print(pilot_symbols_npy[0])
        print(pilot_symbols_2[:100])

        def random_bits(n, seed_no=42):
            np.random.seed(seed_no)
            return np.random.randint(0, 2, size=n, dtype=np.int8)
        print(random_bits(100))
        

        # Get the first 50 frequency blocks (rows) from each array
        npy_first_50 = pilot_symbols_npy[:num_blocks]
        txt_first_50 = pilot_symbols_2[:num_blocks]

        # Check if the two arrays are equal
        are_equal = np.array_equal(npy_first_50, txt_first_50)

        return are_equal
    except Exception as e:
        print(f"Error comparing pilot symbols: {e}")
        return False

compare_pilot_symbols()
