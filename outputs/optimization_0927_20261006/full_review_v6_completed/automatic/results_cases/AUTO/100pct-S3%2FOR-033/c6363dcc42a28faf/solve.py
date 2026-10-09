import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df['Time'])
n_periods = len(period_ids)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_idx_to_time = {i: period_ids[i] for i in range(n_periods)}
time_to_period_idx = {period_ids[i]: i for i in range(n_periods)}
requirement = df['Requirement'].astype(int).to_dict()
requirement_by_idx = {i: requirement[period_ids[i]] for i in range(n_periods)}
shift_length = 16
shift_start_indices = list(range(n_periods))
shift_covers = dict()
for s in shift_start_indices:
    covered = [(s + k) % n_periods for k in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by_shifts = dict()
for t in range(n_periods):
    period_covered_by_shifts[t] = [s for s in shift_start_indices if t in shift_covers[s]]
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
for t in range(n_periods):
    m.addConstr(gp.quicksum((x_vars[s] for s in period_covered_by_shifts[t])) >= requirement_by_idx[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x_vars[s].X for s in shift_start_indices))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        val = int(round(x_vars[s].X))
        if val > 0:
            print(f"  Shift starting at '{period_idx_to_time[s]}': {val} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')