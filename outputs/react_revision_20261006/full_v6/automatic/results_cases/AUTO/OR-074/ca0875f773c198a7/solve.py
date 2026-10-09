import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
expected_cols = {'Time', 'Requirement'}
if set(df.columns) != expected_cols:
    raise ValueError(f'CSV columns {df.columns.tolist()} do not match expected {sorted(expected_cols)}')
if len(df) != 48:
    raise ValueError(f'Expected 48 periods, got {len(df)}')
period_labels = list(df['Time'])
period_indices = list(range(48))
try:
    requirements = {i: int(df.loc[i, 'Requirement']) for i in period_indices}
except Exception as e:
    raise ValueError(f"Failed to parse 'Requirement' as int for all periods: {e}")
shift_length = 16
num_periods = 48

def solve_staffing_minimum():
    m = gp.Model('WaitstaffScheduling')
    shift_start_indices = period_indices
    x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
    for t in period_indices:
        covering_starts = [s for s in shift_start_indices if (t - s) % num_periods < shift_length]
        m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_staffing_minimum()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')