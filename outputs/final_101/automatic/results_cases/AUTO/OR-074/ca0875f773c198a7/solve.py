import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Time', 'Requirement'}.issubset(df.columns):
    raise KeyError("CSV must contain 'Time' and 'Requirement' columns.")
periods = list(df.index)
if len(periods) != 48:
    raise ValueError('Expected 48 periods (rows) in the CSV.')
requirements = df['Requirement'].astype(int).to_dict()
num_periods = 48
shift_length = 16
covering_shifts = {t: [s for s in periods if (t - s) % num_periods < shift_length] for t in periods}
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[s] for s in covering_shifts[t])) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in periods))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in periods:
        staff = int(round(x[s].X))
        if staff > 0:
            print(f"Start at '{df.at[s, 'Time']}': {staff} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')