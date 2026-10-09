import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("CSV must contain 'Time' and 'Requirement' columns.")
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError('Expected 48 time periods (half-hour intervals in 24 hours).')
period_idx = list(range(48))
period_label = dict(zip(period_idx, periods))
requirements = df['Requirement'].astype(int).to_dict()
req = {i: requirements[i] for i in period_idx}
shift_len = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(period_idx, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in period_idx)), gp.GRB.MINIMIZE)
for t in period_idx:
    covering_starts = [s % 48 for s in range(t - shift_len + 1, t + 1)]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= req[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((int(round(x[t].X)) for t in period_idx))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Start Assignments ---')
    for t in period_idx:
        val = int(round(x[t].X))
        if val > 0:
            print(f"  Start at '{period_label[t]}': {val} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')