import numpy as np
import pyscamp as mp

dataset = '1024'

def load_input(name):
    with open(f'data/{name}.txt', 'r') as file:
        data = [*map(float, file.readlines())]
    return np.array(data)

def store_output(name, type, data):
    with open(f'output/{name}-{type}.txt', 'w') as file:
        file.writelines([str(x)+'\n' for x in data])

ts = load_input(dataset)
mp, mpi = mp.selfjoin(ts, 4)

store_output(dataset, 'mp', mp)
store_output(dataset, 'mpi', mpi)