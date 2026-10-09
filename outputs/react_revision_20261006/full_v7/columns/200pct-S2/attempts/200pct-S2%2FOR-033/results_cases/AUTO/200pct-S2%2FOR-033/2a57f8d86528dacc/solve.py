import gurobipy as gp
import pandas as pd
import numpy as np

def solve_shift_scheduling():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others2/44.csv', dtype=str, keep_default_na=False)
    periods = list(range(len(df)))
    if len(periods) != 48:
        raise ValueError(f'Expected 48 periods, got {len(periods)}')
    shift_starts = periods.copy()
    if 'Requirement' not in df.columns:
        raise KeyError("Missing 'Requirement' column in CSV")
    requirement = {}
    for t in periods:
        val = df.loc[t, 'Requirement']
        try:
            requirement[t] = int(val)
        except Exception:
            raise ValueError(f'Invalid Requirement value at period {t}: {val}')
    shift_length = 16
    cover = {}
    for s in shift_starts:
        cover[s] = set(((s + i) % 48 for i in range(shift_length)))
    period_covered_by = {t: [] for t in periods}
    for s in shift_starts:
        for t in cover[s]:
            period_covered_by[t].append(s)
    for t in periods:
        if len(period_covered_by[t]) == 0:
            raise ValueError(f'Period {t} is not covered by any shift start.')
    m = gp.Model('WaitstaffShiftScheduling')
    shift_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((shift_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((shift_vars[s] for s in period_covered_by[t])) >= requirement[t], name=f'cov_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for s in shift_starts:
            print(f'{shift_vars[s].VarName} {shift_vars[s].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_shift_scheduling()