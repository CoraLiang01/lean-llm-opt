import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
time_periods = list(df.index)
shift_starts = list(df.index)
requirement = df['Requirement'].astype(int).to_dict()
periods_per_shift = 16
num_periods = len(time_periods)
shift_coverage = {}
for s in shift_starts:
    s_int = int(s)
    covered = [(s_int + offset) % num_periods for offset in range(periods_per_shift)]
    shift_coverage[s_int] = set(covered)
period_covered_by = {t: set() for t in time_periods}
for s in shift_starts:
    s_int = int(s)
    for t in shift_coverage[s_int]:
        period_covered_by[t].add(s_int)
m = gp.Model('MinWaitstaffShifts')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in time_periods:
    s_covering = period_covered_by[t]
    m.addConstr(gp.quicksum((x_vars[s] for s in s_covering)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule (nonzero only) ---')
    for s in shift_starts:
        val = x_vars[s].X
        if val > 1e-06:
            time_label = df.loc[int(s), 'Time']
            print(f'Shift start at period {s} ({time_label}): {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')