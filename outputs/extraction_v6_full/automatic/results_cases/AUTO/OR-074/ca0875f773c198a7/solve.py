import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("CSV must contain 'Time' and 'Requirement' columns.")
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError('Expected 48 half-hour periods in the day.')
requirements = df['Requirement'].astype(int).to_numpy()
num_periods = 48
shift_length = 16
cover = np.zeros((num_periods, num_periods), dtype=int)
for s in range(num_periods):
    for offset in range(shift_length):
        t = (s + offset) % num_periods
        cover[s, t] = 1
m = gp.Model('MinWaitstaffShifts')
x = m.addVars(num_periods, vtype=gp.GRB.INTEGER, lb=0, name='x')
m.setObjective(gp.quicksum((x[s] for s in range(num_periods))), gp.GRB.MINIMIZE)
for t in range(num_periods):
    m.addConstr(gp.quicksum((x[s] for s in range(num_periods) if cover[s, t])) >= requirements[t], name=f'Coverage_{t}')
m.optimize()