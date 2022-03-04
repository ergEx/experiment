import numpy as np
import json
import os


def main():
    # Constants
    NFRACTALS = 9
    RANDOMSEED = 2022
    ETAS = ['-1.0', '-0.5', '0.0', '0.5', '1.0']

    # Create list of fractal names
    fractal_names = [f'F{i:03d}' for i in range(50)]
    # Create random state generator using the seed
    RS = np.random.RandomState(RANDOMSEED)

    # Create dictionary
    fractal_keys = ([f'{i}' for i in range(1000)] +
                    [chr(i) for i in range(ord('a'), ord('z')+1)])


    fractal_set = {fk : {} for fk in fractal_keys}

    for fk in fractal_keys:
        fractals = RS.choice(fractal_names, len(ETAS) * NFRACTALS, replace=False)

        for n, eta in enumerate(ETAS):
            fractal_set[fk][eta] = list(fractals[n * NFRACTALS : (n + 1) * NFRACTALS])

    with open(f"stimuli{os.sep}fractal_set_nf{NFRACTALS}.json", "w") as outfile:
        json.dump(fractal_set, outfile)

if __name__ == '__main__':
    main()