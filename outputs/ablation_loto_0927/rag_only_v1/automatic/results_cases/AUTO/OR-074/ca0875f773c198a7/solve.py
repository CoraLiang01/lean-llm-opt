import pandas as pd
import numpy as np
from gurobipy import Model, GRB
req_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv', sep=',')
periods = list(req_df['Time'])
if len(periods) != 48:
    raise ValueError('Expected 48 time periods, got {}'.format(len(periods)))
period_idx = list(range(48))
requirements = req_df['Requirement'].astype(int).to_dict()
shift_length = 16
cover = np.zeros((48, 48), dtype=int)
for s in period_idx:
    for i in range(shift_length):
        t = (s + i) % 48
        cover[t, s] = 1
m = Model('Waitstaff_Scheduling')
x = m.addVars(period_idx, vtype=GRB.INTEGER, lb=0, name='')
for t in period_idx:
    m.addConstr(sum((x[s] for s in period_idx if cover[t, s])) >= requirements[t], name=f'cover_{t}')
m.setObjective(sum((x[s] for s in period_idx)), GRB.MINIMIZE)
m.optimize()