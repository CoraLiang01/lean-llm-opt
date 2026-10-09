import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if df.shape[0] != 24:
    raise ValueError(f'Expected 24 rows for 24 hours, got {df.shape[0]} rows.')
df['Shift'] = df['Shift'].astype(int)
shifts = sorted(df['Shift'].unique())
if set(shifts) != set(range(1, 25)):
    raise ValueError(f'Shift indices must be 1..24, got {shifts}')
df['Number Required'] = df['Number Required'].astype(int)
required_staff = {int(row['Shift']): int(row['Number Required']) for (_, row) in df.iterrows()}
shift_indices = list(range(1, 25))

def solve_bus_shift_cover(required_staff, shift_indices, shift_length=4):
    m = gp.Model('BusShiftCover')
    staff_vars = m.addVars(shift_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((staff_vars[t] for t in shift_indices)), gp.GRB.MINIMIZE)
    for s in shift_indices:
        covering_starts = [(s - i - 1) % 24 + 1 for i in range(shift_length)]
        m.addConstr(gp.quicksum((staff_vars[t] for t in covering_starts)) >= required_staff[s], name=f'cover_{s}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_bus_shift_cover(required_staff, shift_indices)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')