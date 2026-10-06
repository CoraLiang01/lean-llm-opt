import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirements = df['Requirement'].astype(int).to_dict()
shift_starts = periods
shift_length = 16
shift_coverage = dict()
for s in shift_starts:
    covered = [(s + i) % n_periods for i in range(shift_length)]
    shift_coverage[s] = set(covered)
period_covered_by = {t: set() for t in periods}
for s in shift_starts:
    for t in shift_coverage[s]:
        period_covered_by[t].add(s)
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_shifts = period_covered_by[t]
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in shift_starts))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule (nonzero only) ---')
    for s in shift_starts:
        val = x[s].X
        if val > 1e-06:
            time_label = df.loc[s, 'Time']
            print(f'Shift start at period {s} ({time_label}): {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')