import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return re.sub('\\s+', '', col).casefold()
col_map = {norm_col(col): col for col in df.columns}
time_col = col_map[norm_col('Time')]
req_col = col_map[norm_col('Requirement')]
periods = list(df[time_col])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
requirements = df[req_col].astype(int).tolist()
if len(requirements) != 48:
    raise ValueError(f'Expected 48 requirements, got {len(requirements)}')
num_periods = 48
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(range(num_periods), vtype=gp.GRB.INTEGER, lb=0, name='')
for t in range(num_periods):
    covering_starts = []
    for s in range(num_periods):
        covered = [(s + offset) % num_periods for offset in range(shift_length)]
        if t in covered:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in range(num_periods))), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((int(round(x_vars[s].X)) for s in range(num_periods)))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('--- Shift Start Assignments ---')
    for s in range(num_periods):
        staff = int(round(x_vars[s].X))
        if staff > 0:
            print(f"  Start at '{periods[s]}': {staff} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')