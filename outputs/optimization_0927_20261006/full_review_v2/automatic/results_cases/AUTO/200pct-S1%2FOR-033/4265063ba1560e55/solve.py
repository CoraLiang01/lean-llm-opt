import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx_to_label = {i: periods[i] for i in range(48)}
period_label_to_idx = {periods[i]: i for i in range(48)}
try:
    requirements = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
period_idx_to_req = {i: int(df.loc[i, 'Requirement']) for i in range(48)}
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(range(48), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in range(48))), gp.GRB.MINIMIZE)
for t in range(48):
    covering_s = [s for s in range(48) if (t - s) % 48 < 16]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_s)) >= period_idx_to_req[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Assignments ---')
    for s in range(48):
        val = x_vars[s].X
        if val > 1e-06:
            print(f'Start at period {s} ({period_idx_to_label[s]}): {int(round(val))} staff')
    print('\n--- Coverage per period ---')
    for t in range(48):
        covering_s = [s for s in range(48) if (t - s) % 48 < 16]
        coverage = sum((x_vars[s].X for s in covering_s))
        print(f'Period {t} ({period_idx_to_label[t]}): Requirement={period_idx_to_req[t]}, Covered={int(round(coverage))}')
else:
    print(f'No optimal solution found. Status: {m.status}')