import json


with open('output.txt', 'r') as file:
    lines: list[dict] = list(map(json.loads, file.readlines()))

csv_data = []

for record in lines:
    for algorithm, details in record.items():
        csv_data.append([algorithm, details['time'], details['moves']])

with open('result.csv', 'w') as csv_file:
    csv_file.write(','.join(['Algorithm', 'Time', 'Moves']) + '\n')
    for line in csv_data:
        csv_file.write(','.join(map(str, line)) + '\n')
