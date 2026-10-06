import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_to_time = dict(zip(periods, df['Time']))
requirement = df['Requirement'].astype(int).to_dict()
shift_starts = periods
num_periods = 48
shift_length = 16
period_covering_shifts = {t: [] for t in periods}
for s in shift_starts:
    covered = [(s + i) % num_periods for i in range(shift_length)]
    for t in covered:
        period_covering_shifts[t].append(s)
for t in periods:
    if len(period_covering_shifts[t]) == 0:
        raise ValueError(f'Period {t} ({period_to_time[t]}) is not covered by any shift.')

def solve_shift_scheduling(periods, shift_starts, requirement, period_covering_shifts):
    m = gp.Model('WaitstaffShiftScheduling')
    m.Params.MIPGap = 0.0001
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[s] for s in period_covering_shifts[t])) >= requirement[t], name=f'cov_{t}')
    m.optimize()
    return m
m = solve_shift_scheduling(periods, shift_starts, requirement, period_covering_shifts)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in shift_starts:
        var = m.getVarByName(f'x[{s}]')
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')