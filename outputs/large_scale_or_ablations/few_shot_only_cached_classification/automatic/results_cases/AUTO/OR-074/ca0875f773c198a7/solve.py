import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(48))
shift_starts = list(range(48))
if len(df) != 48:
    raise ValueError(f'Expected 48 periods in CSV, got {len(df)}')
requirements = df['Requirement'].astype(int).to_dict()
if not all(df.index == np.arange(48)):
    df = df.reset_index(drop=True)
m = gp.Model('MinWaitstaff')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
shift_length = 16
for t in periods:
    covering_starts = [s for s in shift_starts if (t - s) % 48 < shift_length]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_starts:
        if x[s].X > 1e-06:
            time_label = df.loc[s, 'Time'] if s in df.index else f'Period {s}'
            print(f'  Start at {time_label}: {int(round(x[s].X))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')