import os

# Set the path to the original file
original_file = "plaintext/huge.txt"

# Set the desired size of each split file
file_size = 5000

# Open the original file and read its contents
with open(original_file, "r") as f:
    contents = f.read()

# Split the contents into chunks of the desired size
chunks = []
current_chunk = ""
for word in contents.split():
    if len(current_chunk) + len(word) + 1 > file_size:
        chunks.append(current_chunk)
        current_chunk = ""
    current_chunk += word + " "
if current_chunk:
    chunks.append(current_chunk)

# Create a directory to store the split files
if not os.path.exists("plaintext/"):
    os.makedirs("plaintext/")

# Write each chunk to a new file with a unique name
for i, chunk in enumerate(chunks):
    with open(f"plaintext/{original_file.split('/')[-1].split('.')[0]}_{i+1}.txt", "w") as f:
        f.write(chunk)
