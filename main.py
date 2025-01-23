import numpy as np
import math
import numba
import numba.cuda as cuda
import time


T_max = 1000

@cuda.jit(device=True)
def alphaM(V):
    return (2.5 - 0.1 * (V + 65)) / (math.exp(2.5 - 0.1 * (V + 65)) - 1)

@cuda.jit(device=True)
def betaM(V):
    return 4 * math.exp(-(V + 65) / 18)

@cuda.jit(device=True)
def alphaH(V):
    return 0.07 * math.exp(-(V + 65) / 20)

@cuda.jit(device=True)
def betaH(V):
    return 1 / (math.exp(3.0 - 0.1 * (V + 65)) + 1)

@cuda.jit(device=True)
def alphaN(V):
    return (0.1 - 0.01 * (V + 65)) / (math.exp(1 - 0.1 * (V + 65)) - 1)

@cuda.jit(device=True)
def betaN(V):
    return 0.125 * math.exp(-(V + 65) / 80)


@cuda.jit
def HH_gpu(I_values, T, bifurcation_values):
    i = cuda.grid(1)
    if i < I_values.shape[0]:
        I0 = I_values[i]
        dt = 0.01
        T = math.ceil(T / dt)  # [ms]
        gNa0 = 120   # [mS/cm^2]
        ENa = 115   # [mV]
        gK0 = 36    # [mS/cm^2]
        EK = -12    # [mV]
        gL0 = 0.3   # [mS/cm^2]
        EL = 10.6   # [mV]

        V = cuda.local.array(shape=T_max, dtype=numba.float32)
        m = cuda.local.array(T_max, dtype=numba.float32)
        h = cuda.local.array(T_max, dtype=numba.float32)
        n = cuda.local.array(T_max, dtype=numba.float32)

        V[0] = -70.0
        m[0] = 0.05
        h[0] = 0.54
        n[0] = 0.34

        for t in range(T-1):
            V[t+1] = V[t] + dt * (gNa0 * m[t]**3 * h[t] * (ENa - (V[t] + 65)) +
                                  gK0 * n[t]**4 * (EK - (V[t] + 65)) +
                                  gL0 * (EL - (V[t] + 65)) + I0)
            m[t+1] = m[t] + dt * (alphaM(V[t]) * (1 - m[t]) - betaM(V[t]) * m[t])
            h[t+1] = h[t] + dt * (alphaH(V[t]) * (1 - h[t]) - betaH(V[t]) * h[t])
            n[t+1] = n[t] + dt * (alphaN(V[t]) * (1 - n[t]) - betaN(V[t]) * n[t])

        # Perform a reduction to find the maximum value
        max_value = V[0]
        for t in range(T):
            if V[t] > max_value:
                max_value = V[t]

        min_value = V[0]
        for t in range(T):
            if V[t] < min_value:
                min_value = V[t]

        bifurcation_values[i, 0] = max_value
        bifurcation_values[i, 1] = min_value


def compute_bifurcation_gpu(I_values, T):
    bifurcation_values = np.zeros((I_values.shape[0], 2), dtype=np.float32)
    threads_per_block = 128
    blocks_per_grid = math.ceil(I_values.shape[0] / threads_per_block)
    HH_gpu[blocks_per_grid, threads_per_block](I_values, T, bifurcation_values)
    return bifurcation_values

print(cuda.gpus)
# Define the range of applied current values and simulation duration
I_values = np.linspace(0, 180, 1000).astype(np.float32)  # Range of applied current values
T = 100  # Simulation duration in milliseconds

# Measure execution time
start_time = time.time()

# Compute the bifurcation values on the GPU
bifurcation_values = compute_bifurcation_gpu(I_values, T)

end_time = time.time()
execution_time = end_time - start_time

print("Execution time: {:.2f} seconds".format(execution_time))