import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = df['Time'].tolist()
if len(period_ids) != 48:
    raise ValueError(f'Expected 48 periods, got {len(period_ids)}')
requirement = df['Requirement'].astype(int).to_dict()
period_idx_to_id = {i: period_ids[i] for i in range(48)}
period_id_to_idx = {period_ids[i]: i for i in range(48)}
T = list(range(48))
S = list(range(48))
m = gp.Model('MinWaitstaffShifts')
x_vars = m.addVars(S, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in S)), gp.GRB.MINIMIZE)
shift_length = 16
for t in T:
    covering_s = [s for s in S if (t - s) % 48 < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_s)) >= requirement[str(t)], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((int(round(x_vars[s].X)) for s in S))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in S:
        num = int(round(x_vars[s].X))
        if num > 0:
            print(f"  Start at '{period_idx_to_id[s]}': {num} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')