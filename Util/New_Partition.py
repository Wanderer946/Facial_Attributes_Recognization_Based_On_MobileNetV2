def process_partition_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        filename, _ = line.strip().split()
        if '000001.jpg' <= filename <= '003000.jpg':
            index = int(filename[:6])
            if index <= 2400:
                partition = 0  # train
            elif index <= 2700:
                partition = 1  # val
            else:
                partition = 2  # test
            new_lines.append(f"{filename} {partition}\n")

    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Saved success, Totally {len(new_lines)} Saved to {output_path}")

if __name__ == "__main__":
    process_partition_file('Dataset/list_partition.txt', 'Dataset/list_partition_3000.txt')
