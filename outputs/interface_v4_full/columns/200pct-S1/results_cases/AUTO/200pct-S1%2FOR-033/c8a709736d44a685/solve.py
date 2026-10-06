import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if 'Requirement' not in df.columns or 'Time' not in df.columns:
    raise KeyError("CSV must contain 'Requirement' and 'Time' columns.")
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirement = df['Requirement'].astype(int).to_dict()
time_labels = df['Time'].astype(str).to_dict()
shift_length = 16
shift_covers = dict()
for s in periods:
    covered = [(s + i) % n_periods for i in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = {t: set() for t in periods}
for s in periods:
    for t in shift_covers[s]:
        period_covered_by[t].add(s)
m = gp.Model('Waitstaff_Scheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[s] for s in period_covered_by[t])) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum number of waitstaff)')
    print('\n--- Shift Start Assignments ---')
    for s in periods:
        val = x[s].X
        if val > 0.5:
            print(f'  Start at period {s:2d} ({time_labels[s]}): {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')