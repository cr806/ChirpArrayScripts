import json

from pathlib import Path

num_of_files = 14

input_json_path = Path('/Volumes/krauss/Lisa/GMR/Array/250225/loc1_1/Split')
data_json_name = Path('ROI_SU000001.json')

output_path = input_json_path
output_path.mkdir(parents=True, exist_ok=True)

with Path(input_json_path, data_json_name).open('r') as f:
    data = json.load(f)

items = list(data.items())
total = len(items)
print(f'Number of dictionaries: {total}')

base_size = total // num_of_files
remainder = total % num_of_files
split_sizes = [base_size + (1 if i < remainder else 0)
               for i in range(num_of_files)]

# Split the items into three chunks
chunks = []
start = 0
for size in split_sizes:
    end = start + size
    chunks.append(dict(items[start:end]))
    start = end

# Save each chunk to a separate JSON file
for i, chunk in enumerate(chunks, 1):
    output_file = Path(output_path, f'{data_json_name.stem}_{i}.json')
    with output_file.open('w') as f:
        json.dump(chunk, f, indent=4)
    print(f'Saved {len(chunk)} dictionaries to {output_file}')