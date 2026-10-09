import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df['Time'])
if len(period_ids) != 48:
    raise ValueError(f'Expected 48 periods, got {len(period_ids)}')
period_idx_to_id = {i: period_ids[i] for i in range(48)}
period_id_to_idx = {period_ids[i]: i for i in range(48)}
try:
    requirement_per_period = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
requirement = {i: int(df.loc[i, 'Requirement']) for i in range(48)}
S = list(range(48))
T = list(range(48))
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(S, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in T:
    covering_starts = [s for s in S if (t - s) % 48 in range(shift_length)]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in S)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in S:
        num = x_vars[s].X
        if num > 1e-06:
            print(f'  Shift starting at period {s} ({period_idx_to_id[s]}): {int(round(num))} waitstaff')
    print('\n--- Coverage per period ---')
    for t in T:
        covering_starts = [s for s in S if (t - s) % 48 in range(shift_length)]
        coverage = sum((x_vars[s].X for s in covering_starts))
        print(f'  Period {t} ({period_idx_to_id[t]}): Requirement={requirement[t]}, Covered={int(round(coverage))}')
else:
    print(f'No optimal solution found. Status: {m.status}')