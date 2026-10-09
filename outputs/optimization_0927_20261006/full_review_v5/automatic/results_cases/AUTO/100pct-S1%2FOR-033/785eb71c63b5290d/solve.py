import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df.index)
num_periods = len(period_ids)
if num_periods != 48:
    raise ValueError(f'Expected 48 periods, got {num_periods}')
requirement = {}
for idx in period_ids:
    req_str = df.loc[idx, 'Requirement']
    try:
        req = int(req_str)
    except Exception:
        raise ValueError(f'Invalid Requirement value at row {idx}: {req_str}')
    requirement[idx] = req
shift_starts = period_ids
shift_length = 16
shift_covers = {}
for s in shift_starts:
    covered = [(s + offset) % num_periods for offset in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = {t: [] for t in period_ids}
for s in shift_starts:
    for t in shift_covers[s]:
        period_covered_by[t].append(s)
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in period_ids:
    m.addConstr(gp.quicksum((x_vars[s] for s in period_covered_by[t])) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x_vars[s].X)) for s in shift_starts))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule (nonzero only) ---')
    for s in shift_starts:
        val = int(round(x_vars[s].X))
        if val > 0:
            time_label = df.loc[s, 'Time']
            print(f'  Start at period {s} ({time_label}): {val} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')