import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
requirement = {}
for (idx, row) in df.iterrows():
    period = row['Time']
    try:
        req = int(row['Requirement'])
    except Exception as e:
        raise ValueError(f"Invalid Requirement value at period '{period}': {row['Requirement']}") from e
    requirement[period] = req
shift_starts = periods.copy()
num_periods = len(periods)
shift_length = 16
period_to_idx = {period: idx for (idx, period) in enumerate(periods)}
idx_to_period = {idx: period for (idx, period) in enumerate(periods)}
shift_coverage = dict()
for (s_idx, s) in enumerate(shift_starts):
    covered_indices = [(s_idx + offset) % num_periods for offset in range(shift_length)]
    covered_periods = [idx_to_period[i] for i in covered_indices]
    shift_coverage[s] = set(covered_periods)
period_covered_by_shifts = {t: set() for t in periods}
for s in shift_starts:
    for t in shift_coverage[s]:
        period_covered_by_shifts[t].add(s)
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_shifts = period_covered_by_shifts[t]
    if not covering_shifts:
        raise ValueError(f"No shift covers period '{t}'")
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_starts:
        val = x_vars[s].X
        if val > 1e-06:
            print(f'Shift starting at {s}: {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')