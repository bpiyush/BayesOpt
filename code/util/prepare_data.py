import numpy as np

def f(x):
    return np.sinc(x * 10 - 5)

def get_toy_dataset(num_points=20):
    # Fix the random state
    rng = np.random.RandomState(42)

    x = rng.rand(num_points)
    y = f(x)
    
    toy_dataset = x, y
    return toy_dataset