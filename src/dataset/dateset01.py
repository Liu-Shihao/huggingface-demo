from datasets import load_dataset
dataset = load_dataset("text", data_files={"train": ["my_text_1.txt", "my_text_2.txt"], "test": "my_test_file.txt"})

dataset = load_dataset("text", data_dir="path/to/text/dataset")