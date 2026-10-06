import gurobipy as gp
import pandas as pd
import numpy as np

def solve_min_waitstaff():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
    df = pd.read_csv(csv_path, sep=',')
    periods = list(df.index)
    n_periods = len(periods)
    if n_periods != 48:
        raise ValueError(f'Expected 48 periods, got {n_periods}')
    requirement = df['Requirement'].astype(int).to_dict()
    shift_length = 16
    m = gp.Model('MinWaitstaff')
    x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[t] for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        covering_starts = [(t - offset) % n_periods for offset in range(shift_length)]
        m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
    m.optimize()
    return m
m = solve_min_waitstaff()