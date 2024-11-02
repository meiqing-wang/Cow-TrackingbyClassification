import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--inp', type=str, help='txt file')
parser.add_argument('--out', type=str, help='txt file')
args = parser.parse_args()

# Define the input and output file paths
input_file_path  = args.inp
output_file_path = args.out

# Read the content of the file
with open(input_file_path, 'r') as file:
    lines = file.read().splitlines()
    #print(lines[:5])

# Filter out lines with objectID equal to 14
lines = [line for line in lines if line.strip().split(',')[1] != '14']
    
# Modify objectID 2 to 5
for i in range(len(lines)):
    parts = lines[i].strip().split(',')
    if parts[1] == '2':
        parts[1] = '5'
    #elif 3 <= int(parts[1]) <= 14:
    #    parts[1] = str(int(parts[1]) - 1)

    # Update the line with modified objectID
    lines[i] = ','.join(parts) + '\n'

# Write the updated content back to the file
with open(output_file_path, 'w') as file:
    file.writelines(lines)

print("File updated successfully!")
