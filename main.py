import json

with open('./data/paired_dataset.json') as f:
    data = json.load(f)

for item in data:
    item['image_path'] = './data/' + item['image_path']

with open('./data/paired_dataset.json', 'w') as f:
    json.dump(data, f, indent=2)

print('Paths updated')