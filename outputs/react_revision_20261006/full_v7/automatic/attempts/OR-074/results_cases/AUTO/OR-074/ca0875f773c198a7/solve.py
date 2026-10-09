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
periods = list(range(48))
period_idx_to_label = dict(zip(periods, df['Time']))
if not df['Requirement'].apply(lambda x: re.fullmatch('\\d+', x)).all():
    raise ValueError("Non-integer or missing values found in 'Requirement' column")
requirement = {t: int(df.iloc[t]['Requirement']) for t in periods}
shift_starts = periods.copy()
shift_length = 16
covered_periods_by_shift = {s: [(s + i) % 48 for i in range(shift_length)] for s in shift_starts}
covering_shifts_by_period = {t: [] for t in periods}
for s in shift_starts:
    for t in covered_periods_by_shift[s]:
        covering_shifts_by_period[t].append(s)
for t in periods:
    if len(covering_shifts_by_period[t]) == 0:
        raise ValueError(f'Period {t} ({period_idx_to_label[t]}) is not covered by any shift.')

def solve_minimum_waitstaff(shift_starts, periods, requirement, covering_shifts_by_period):
    m = gp.Model('MinimumWaitstaff')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts_by_period[t])) >= requirement[t], name=f'cov{t}')
    m.optimize()
    return m
m = solve_minimum_waitstaff(shift_starts, periods, requirement, covering_shifts_by_period)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in shift_starts:
        var = m.getVarByName(f'x[{s}]')
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')