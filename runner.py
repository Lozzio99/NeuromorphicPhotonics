from email.encoders import encode_base64

import numpy as np
from numpy import multiply, add
import config
from laser import LaserSystem


ITERATIONS_LOG = 5e4

def euler_mayurama_step(laser:LaserSystem, t:float, h:float):
    # 2 steps numerical integration solver, includes noise
    noise = laser.noise_f()                                 # 3D gaussian noise
    s0 = laser.get_sysvar()                                 # state y(t)
    fs0 = laser.rate_equations(s0, t)                       # rate equations for y(t)
    s1 = add(add(s0, multiply(h, fs0)), noise)              # first step new state y*(t)
    fs1 = laser.rate_equations(s1, t)                       # rate equations for y*(t)
    s2 = add(add(s0, multiply(h/2, add(fs0, fs1))), noise)  # second step new state y(t+h)
    return s2


def run_single_laser_simulation(laser:LaserSystem, t0=config.t0, tf=config.tf, dt=config.dt):
    t = t0
    time_sequence = np.arange(t0, tf, dt)
    solution = {'t': [t]}
    state0 = laser.get_state_dict()
    for var_name, var_value in state0.items(): solution[var_name] = [var_value]

    i = 1
    while i < len(time_sequence) and t <= tf:
        if i % ITERATIONS_LOG == 0:
            print(f"Iteration {i}/{len(time_sequence)}")

        t1 = time_sequence[i]
        h = t1 - t

        step_update_state = euler_mayurama_step(laser, t, h)  # step update variables
        laser.update(step_update_state, t1)                   # update laser to new state and time

        new_state = laser.get_state_dict()
        solution['t'].append(t1)
        for var_name, var_value in new_state.items(): solution[var_name].append(var_value)

        t = t1
        i+=1

    return solution
