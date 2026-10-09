import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv', dtype=str, keep_default_na=False)
df['Requirement'] = df['Requirement'].astype(int)
period_ids = list(df.index)
shift_start_ids = list(df.index)
requirement = df['Requirement'].to_dict()
periods_per_shift = 16
shift_coverage = {}
for s in shift_start_ids:
    s_int = int(s)
    covered = [(s_int + i) % len(period_ids) for i in range(periods_per_shift)]
    shift_coverage[s_int] = set(covered)
period_covered_by_shifts = {t: set() for t in period_ids}
for (s_int, covered_set) in shift_coverage.items():
    for t in covered_set:
        period_covered_by_shifts[t].add(s_int)
for t in period_ids:
    if not period_covered_by_shifts[t]:
        raise ValueError(f'Period {t} is not covered by any shift.')
m = Model('Waitstaff_Scheduling')
m.Params.OutputFlag = 0
x_vars = m.addVars(shift_start_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((x_vars[s] for s in shift_start_ids)), GRB.MINIMIZE)
for t in period_ids:
    m.addConstr(quicksum((x_vars[s] for s in period_covered_by_shifts[t])) >= requirement[t], name=f'cover_{t}')
m.optimize()