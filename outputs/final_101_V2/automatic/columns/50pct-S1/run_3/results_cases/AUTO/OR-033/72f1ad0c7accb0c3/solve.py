import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_labels = df['Time'].tolist()
requirement = df['Requirement'].astype(int).to_dict()
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covered_starts = []
    for s in periods:
        if (t - s) % n_periods < shift_length:
            covered_starts.append(s)
    if not covered_starts:
        raise ValueError(f'No shift starts cover period {t}')
    m.addConstr(gp.quicksum((x[s] for s in covered_starts)) >= requirement[t], name=f'cov_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Assignments ---')
    for s in periods:
        val = x[s].X
        if val > 1e-06:
            print(f'  Shift starting at period {s} ({period_labels[s]}): {int(round(val))} waitstaff')
    print('\n--- Coverage per period (for verification) ---')
    staff_on_duty = []
    for t in periods:
        total = 0
        for s in periods:
            if (t - s) % n_periods < shift_length:
                total += x[s].X
        staff_on_duty.append(total)
        print(f'  Period {t} ({period_labels[t]}): Requirement={requirement[t]}, On duty={int(round(total))}')
else:
    print(f'No optimal solution found. Status: {m.status}')