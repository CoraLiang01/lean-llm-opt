import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df.index)
num_periods = len(period_ids)
shift_length = 16
requirement = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(period_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in period_ids)), gp.GRB.MINIMIZE)
for t in period_ids:
    covering_shifts = []
    for s in period_ids:
        covered = [(s + offset) % num_periods for offset in range(shift_length)]
        if t in covered:
            covering_shifts.append(s)
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()