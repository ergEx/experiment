import numpy as np
import json
import os


def main():
    # Constants
    NFRACTALS = 9
    RANDOMSEED = 2022
    RANDOMSEED2 = 2023
    SESSIONS = ['1', '2' , '3', '4', '5']

    # Create list of fractal names
    fractal_names = [f'F{i:03d}' for i in range(50)]
    # Create random state generator using the seed
    RS = np.random.RandomState(RANDOMSEED)
    RS2 = np.random.RandomState(RANDOMSEED2)

    # Create dictionary
    fractal_keys = ([f'{i}' for i in range(1000)] +
                    [chr(i) for i in range(ord('a'), ord('z')+1)])


    fractal_set = {fk : {} for fk in fractal_keys}

    for fk in fractal_keys:
        fractals = RS.choice(fractal_names, len(SESSIONS) * NFRACTALS, replace=False)
        lambds = np.hstack([RS.choice(['0.0', '1.0'], 2, replace=False) for _ in range(3)])

        for n, (sess, lmb) in enumerate(zip(SESSIONS, lambds)):
            fractal_set[fk][sess] = [lmb, list(fractals[n * NFRACTALS : (n + 1) * NFRACTALS])]

    with open(f"stimuli{os.sep}fractal_set_nf{NFRACTALS}.json", "w") as outfile:
        json.dump(fractal_set, outfile)

if __name__ == '__main__':
    main()