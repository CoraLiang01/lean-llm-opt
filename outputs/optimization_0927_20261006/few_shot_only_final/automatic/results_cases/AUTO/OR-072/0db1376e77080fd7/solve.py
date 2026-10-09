import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if df.shape[0] != 24:
    raise ValueError(f'Expected 24 rows for 24 hours, got {df.shape[0]} rows.')
if not all(df['Number Required'].apply(lambda x: re.fullmatch('\\d+', x))):
    raise ValueError("Non-numeric or missing values found in 'Number Required' column.")
number_required = {t + 1: int(df.iloc[t]['Number Required']) for t in range(24)}
m = gp.Model('BusRouteStaffScheduling')
x_vars = m.addVars(range(1, 25), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in range(1, 25))), gp.GRB.MINIMIZE)
for h in range(1, 25):
    covered_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[tau] for tau in covered_starts)) >= number_required[h], name=f'cover_{h}')
m.optimize()