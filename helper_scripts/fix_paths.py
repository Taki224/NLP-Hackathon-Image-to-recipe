import json

with open('/home/s22imc10262/data/NLP_hackathon_data/paired_dataset.json') as f:
    data = json.load(f)

for item in data:
    item['image_path'] = item['image_path'].replace(
        './data/', '/home/s22imc10262/data/NLP_hackathon_data/'
    )

with open('/home/s22imc10262/data/NLP_hackathon_data/paired_dataset.json', 'w') as f:
    json.dump(data, f, indent=2)

print('Paths updated')