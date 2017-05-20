# List all file in the dataset directory
all_file = os.listdir("dataset")
# Filter : Select only c file
all_file_name = np.array([f for f in all_file if f.find(".c") != -1])

content = ""
for name in all_file_name:
    with open(os.path.join("dataset", name), "r") as f:
        content += f.read() + "\n"

# Convert the string into a list of interger
vocab = set(content)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in content], dtype=np.int32)
