import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx_to_id = {i: periods[i] for i in range(48)}
period_id_to_idx = {periods[i]: i for i in range(48)}
try:
    requirement_per_period = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
requirement = {i: int(df.loc[i, 'Requirement']) for i in range(48)}
m = gp.Model('WaitstaffScheduling')
shift_start_vars = m.addVars(range(48), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s] for s in range(48))), gp.GRB.MINIMIZE)
shift_length = 16
for t in range(48):
    covering_starts = [s % 48 for s in range(t - shift_length + 1, t + 1)]
    m.addConstr(gp.quicksum((shift_start_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in range(48):
        n = shift_start_vars[s].X
        if n > 1e-06:
            print(f"  Start at '{period_idx_to_id[s]}': {int(round(n))} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')