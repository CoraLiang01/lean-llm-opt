import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in {csv_path}")
periods = list(df.index)
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods (rows), got {len(periods)}')
period_labels = df['Time'].tolist()
try:
    requirements = [int(x) for x in df['Requirement']]
except Exception as e:
    raise ValueError(f"Non-integer value in 'Requirement' column: {e}")
shift_length = 16
num_periods = 48
shift_start_indices = list(range(num_periods))

def solve_shift_scheduling():
    m = gp.Model('WaitstaffShiftScheduling')
    shift_start_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((shift_start_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
    for t in range(num_periods):
        covering_starts = []
        for s in shift_start_indices:
            if 0 <= (t - s) % num_periods < shift_length:
                covering_starts.append(s)
        if not covering_starts:
            raise ValueError(f'No shift start covers period {t}')
        m.addConstr(gp.quicksum((shift_start_vars[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_shift_scheduling()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')