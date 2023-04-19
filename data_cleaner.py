input_file = "./trunk_data/spo_train_multiword.txt"
output_file = "./learning_data/spo_train_cleaned_multiword.txt"

with open(input_file) as ifile:
    with open(output_file, "w") as ofile:
        for line in ifile:
            if "[]" not in line:
                ofile.write(line)
