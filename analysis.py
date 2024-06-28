import json


with open('output.txt', 'r') as file:
    lines: list[dict] = list(map(json.loads, file.readlines()))

csv_data = set()

algorithms = [
    'a_star_hamming',
    'a_star_manhattan',
    'greedy_hamming',
    'greedy_manhattan',
    'bfs',
    'dfs',
]

for record in lines:
    for keys in record.keys():
        for algorithm in algorithms:
            csv_data.add(','.join(map(str, [
                f'"{"".join(map(lambda x: "".join(map(str, x)), record["input"]))}"',
                f'"{algorithm}"',
                f'"{record[algorithm]["time"]}"',
                f'"{record[algorithm]["moves"]}"',
                f'"{record[algorithm]["visited"]}"',
            ])))

with open('result.csv', 'w') as csv_file:
    csv_file.write(','.join(map(lambda x: f'"{x}"', [
        'Input',
        'Algorithm',
        'Time',
        'Moves',
        'Visited',
    ])) + '\n')
    for line in csv_data:
        csv_file.write(line + '\n')
