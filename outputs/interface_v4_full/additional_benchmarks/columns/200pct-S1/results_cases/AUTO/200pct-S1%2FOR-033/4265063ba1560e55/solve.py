import gurobipy as gp
import pandas as pd
import numpy as np

def solve_shift_scheduling():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
    df = pd.read_csv(csv_path, sep=',')
    periods = list(df.index)
    n_periods = len(periods)
    if n_periods != 48:
        raise ValueError(f'Expected 48 periods, got {n_periods}')
    shift_starts = periods.copy()
    requirements = df['Requirement'].astype(int).to_dict()
    m = gp.Model('WaitstaffScheduling')
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    shift_length = 16
    for t in periods:
        covering_starts = [s for s in shift_starts if (t - s) % n_periods in range(shift_length)]
        m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
    m.optimize()
    return m
m = solve_shift_scheduling()