import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Time', 'Requirement'}.issubset(df.columns):
    raise ValueError("CSV must contain 'Time' and 'Requirement' columns.")
time_periods = list(df['Time'])
n_periods = len(time_periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 time periods, got {n_periods}.')
requirements = df['Requirement'].astype(int).tolist()
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(n_periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in range(n_periods))), gp.GRB.MINIMIZE)
for t in range(n_periods):
    covering_starts = [s for s in range(n_periods) if (t - s) % n_periods < shift_length]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()