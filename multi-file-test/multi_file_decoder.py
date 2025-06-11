import numpy as np

byte_array = np.packbits(np.load("multi-file-test/examples.npy"))
print("Byte array shape: ", byte_array.shape)

current_offset = 0
file_count = 1
    
while current_offset < byte_array.size:
    print(f"\n--- Attempting to decode file {file_count} starting at offset {current_offset} ---")
    try:
        # 1. Find the end of the filename (first null terminator)
        first_null_search_area = byte_array[current_offset:]
        null_indices = np.where(first_null_search_area == 0)[0]
        
        if null_indices.size < 2:
            print("Error: Not enough null terminators found for a full file entry. Stopping.")
            break
        
        name_terminator_rel = null_indices[0]
        name_terminator_abs = current_offset + name_terminator_rel
        
        # 2. Decode the filename
        file_name_bytes = byte_array[current_offset:name_terminator_abs]
        file_name = file_name_bytes.tobytes().decode("utf-8")
        print(f"Found filename: '{file_name}'")

        # 3. Find the end of the file size (second null terminator)
        size_terminator_rel = null_indices[1]
        size_terminator_abs = current_offset + size_terminator_rel
        
        # 4. Decode the file size
        file_size_bytes = byte_array[name_terminator_abs + 1:size_terminator_abs]
        file_size_str = file_size_bytes.tobytes().decode("utf-8")
        
        if not file_size_str.isdigit():
            print(f"Error: Decoded file size '{file_size_str}' is not a valid number. Stopping.")
            break
        file_size = int(file_size_str)
        print(f"Found file size: {file_size} bytes")

        # 5. Extract the file content
        content_start_abs = size_terminator_abs + 1
        content_end_abs = content_start_abs + file_size
        
        if content_end_abs > byte_array.size:
            print(f"Error: Required file content (ends at {content_end_abs}) exceeds byte array size ({byte_array.size}). Stopping.")
            break
            
        file_content = byte_array[content_start_abs:content_end_abs]
        
        # 6. Determine the correct file extension and write the file
        if file_name.endswith(".tiff") or file_name.endswith(".tif"):
            output_filename = f"Decoded_File_{file_count}.tiff"
        elif file_name.endswith(".wav"):
            output_filename = f"Decoded_File_{file_count}.wav"
        elif file_name.endswith(".txt"):
            output_filename = f"Decoded_File_{file_count}.txt"
        elif file_name.endswith(".bmp"):
            output_filename = f"Decoded_File_{file_count}.bmp"
        elif file_name.endswith(".html"):
            output_filename = f"Decoded_File_{file_count}.html"
        else:
            output_filename = f"Decoded_File_{file_count}.dat"

        with open(output_filename, "wb") as f:
            f.write(file_content.tobytes())
        print(f"SUCCESS: Saved decoded file as '{output_filename}'")

        # 7. Update offset for the next file and increment counter
        current_offset = content_end_abs
        file_count += 1

    except (IndexError, ValueError, UnicodeDecodeError) as e:
        print(f"\n>>> An error occurred while parsing file {file_count}: {e}")
        print(">>> The byte stream may be corrupted. Halting decode process.")
        print(f">>> Last processed offset was {current_offset}.")
        error_region = byte_array[max(0, current_offset-16):current_offset+64]
        print(f">>> Bytes around error point: {error_region}")
        break

print("\n--- Multi-file decoding finished ---")