import random


def generate_initial_states(num_states, puzzle_size):
    states = set()
    while len(states) < num_states:
        state = random.sample(range(puzzle_size), puzzle_size)
        state_str = ','.join(map(str, state))
        states.add(state_str)
    return list(states)


num_states = 10
puzzle_size = 9

initial_states = generate_initial_states(num_states, puzzle_size)

with open('input.txt', 'w') as f:
    for state in initial_states:
        f.write(f'{state}\n')
