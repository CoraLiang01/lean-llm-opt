import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Requirement' not in df.columns:
    raise KeyError("Required column 'Requirement' not found in 44.csv")
if len(df) != 48:
    raise ValueError('Expected 48 periods in 44.csv, got {}'.format(len(df)))
periods = list(range(48))
requirements = {}
for idx, row in df.iterrows():
    try:
        requirements[idx] = int(row['Requirement'])
    except Exception as e:
        raise ValueError(f"Invalid Requirement value at row {idx}: {row['Requirement']}") from e
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in periods)), gp.GRB.MINIMIZE)
for p in periods:
    covering_starts = [t for t in periods if (p - t) % 48 < shift_length]
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_starts)) >= requirements[p], name=f'cover_{p}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for t in periods:
        val = x_vars[t].X
        if val > 0.5:
            time_label = df.loc[t, 'Time'] if 'Time' in df.columns else f'Period {t}'
            print(f'  Start at {time_label}: {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')