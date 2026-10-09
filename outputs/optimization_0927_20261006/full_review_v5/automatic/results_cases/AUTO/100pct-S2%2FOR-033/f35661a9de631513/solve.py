import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
shift_starts = periods.copy()
period_idx_to_time = {i: periods[i] for i in range(48)}
time_to_period_idx = {periods[i]: i for i in range(48)}
try:
    requirement_per_period = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
requirement = {i: int(df.loc[i, 'Requirement']) for i in range(48)}
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(range(48), vtype=gp.GRB.INTEGER, lb=0, name='')
for p in range(48):
    covering_shift_starts = []
    for s in range(48):
        covered_periods = [(s + offset) % 48 for offset in range(16)]
        if p in covered_periods:
            covering_shift_starts.append(s)
    if not covering_shift_starts:
        raise ValueError(f'No shift covers period {p} ({period_idx_to_time[p]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shift_starts)) >= requirement[p], name=f'cover_{p}')
m.setObjective(gp.quicksum((x_vars[s] for s in range(48))), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in range(48):
        val = x_vars[s].X
        if val > 0.5:
            print(f"  Shift starting at '{period_idx_to_time[s]}': {int(round(val))} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')