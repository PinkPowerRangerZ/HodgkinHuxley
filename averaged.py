import numpy as np
import math
import time
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt

def alphaM(V):
    return (2.5 - 0.1 * (V + 65)) / (np.exp(2.5 - 0.1 * (V + 65)) - 1)

def betaM(V):
    return 4 * np.exp(-(V + 65) / 18)

def alphaH(V):
    return 0.07 * np.exp(-(V + 65) / 20)

def betaH(V):
    return 1 / (np.exp(3.0 - 0.1 * (V + 65)) + 1)

def alphaN(V):
    return (0.1 - 0.01 * (V + 65)) / (np.exp(1 - 0.1 * (V + 65)) - 1)

def betaN(V):
    return 0.125 * np.exp(-(V + 65) / 80)

def HH(params):
    I0, T0 = params
    dt = 0.01
    T = math.ceil(T0 / dt)  # [ms]
    gNa0 = 120   # [mS/cm^2]
    ENa = 115   # [mV]
    gK0 = 36    # [mS/cm^2]
    EK = -12    # [mV]
    gL0 = 0.3   # [mS/cm^2]
    EL = 10.6   # [mV]

    t = np.arange(0, T) * dt
    V = np.zeros(T)
    m = np.zeros(T)
    h = np.zeros(T)
    n = np.zeros(T)

    V[0] = -70.0
    m[0] = 0.05
    h[0] = 0.54
    n[0] = 0.34

    for i in range(T-1):
        V[i+1] = V[i] + dt * (gNa0 * m[i]**3 * h[i] * (ENa - (V[i] + 65)) +
                              gK0 * n[i]**4 * (EK - (V[i] + 65)) +
                              gL0 * (EL - (V[i] + 65)) + I0)
        m[i+1] = m[i] + dt * (alphaM(V[i]) * (1 - m[i]) - betaM(V[i]) * m[i])
        h[i+1] = h[i] + dt * (alphaH(V[i]) * (1 - h[i]) - betaH(V[i]) * h[i])
        n[i+1] = n[i] + dt * (alphaN(V[i]) * (1 - n[i]) - betaN(V[i]) * n[i])

    return max(V[-1000:]), min(V[-1000:])


def compute_bifurcation_parallel(I_values, T):
    bifurcation_values = []
    with Pool() as pool:
        params = [(I, T) for I in I_values]
        bifurcation_values = pool.map(HH, params)
    return bifurcation_values


def compute_bifurcation_sequential(I_values, T):
    bifurcation_values = []
    for I in I_values:
        params = (I, T)
        bifurcation_values.append(HH(params))
    return bifurcation_values


def compare_running_times(I_values, T):
    parallel_execution_times = []
    sequential_execution_times = []

    for _ in range(10):
        start_time = time.time()
        bifurcation_values_parallel = compute_bifurcation_parallel(I_values, T)
        end_time_parallel = time.time()
        execution_time_parallel = end_time_parallel - start_time
        parallel_execution_times.append(execution_time_parallel)

        start_time = time.time()
        bifurcation_values_sequential = compute_bifurcation_sequential(I_values, T)
        end_time_sequential = time.time()
        execution_time_sequential = end_time_sequential - start_time
        sequential_execution_times.append(execution_time_sequential)

    return parallel_execution_times, sequential_execution_times


if __name__ == '__main__':
    # Define the range of applied current values and simulation duration
    I_values_range = [(0, 80), (0, 180), (0, 220)]  # Different ranges of applied current values
    T = 100  # Simulation duration in milliseconds

    table_data = []

    for I_range in I_values_range:
        I_values = np.linspace(I_range[0], I_range[1], 1000)

        parallel_times, sequential_times = compare_running_times(I_values, T)

        avg_parallel_time = np.mean(parallel_times)
        avg_sequential_time = np.mean(sequential_times)

        table_data.append([I_range[0], I_range[1], avg_parallel_time, avg_sequential_time])

    table = pd.DataFrame(table_data, columns=['I_values_start', 'I_values_end', 'Parallel Time', 'Sequential Time'])

    print(table)
    table.to_csv("intervals_1.csv")


