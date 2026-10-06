import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(48))
if len(df) != 48:
    raise ValueError(f'Expected 48 periods in the CSV, found {len(df)}')
requirements = df['Requirement'].astype(int).to_dict()
shift_starts = periods.copy()
shift_covers = dict()
for s in shift_starts:
    covered = [(s + i) % 48 for i in range(16)]
    shift_covers[s] = set(covered)
period_covered_by = {t: [] for t in periods}
for s in shift_starts:
    for t in shift_covers[s]:
        period_covered_by[t].append(s)
m = gp.Model('MinWaitstaffShifts')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='x')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[s] for s in period_covered_by[t])) >= requirements[t], name=f'Coverage_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((x[s].X for s in shift_starts))
    print(f'Optimal total value/cost: {total_waitstaff:.0f} (minimum number of waitstaff)')
    print('--- Shift assignments (start period : number of staff) ---')
    for s in shift_starts:
        if x[s].X > 1e-06:
            time_label = df.iloc[s]['Time']
            print(f'  Start at period {s:2d} ({time_label}): {int(round(x[s].X))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')