import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',')
worker_list = [col for col in df.columns if col != 'Owner']
homeowner_list = df['Owner'].astype(str).tolist()
if set(homeowner_list) != set(worker_list):
    raise ValueError('Mismatch between homeowners and workers. Each homeowner must correspond to a worker.')
work_days = {}
for i, row in df.iterrows():
    owner = str(row['Owner'])
    work_days[owner] = {}
    for worker in worker_list:
        work_days[owner][worker] = int(row[worker])
for worker in worker_list:
    total_days = sum((work_days[owner][worker] for owner in homeowner_list))
    if total_days != 10:
        raise ValueError(f'Worker {worker} has total work days {total_days}, expected 10.')
m = Model('mutual_wage_payment')
m.Params.OutputFlag = 0
w = {}
for worker in worker_list:
    if worker == worker_list[0]:
        w[worker] = m.addVar(lb=60.0, ub=60.0, vtype=GRB.CONTINUOUS, name='w_fixed')
    else:
        w[worker] = m.addVar(vtype=GRB.CONTINUOUS, name='w_' + worker)
m.update()
for k in worker_list:
    income = sum((work_days[owner][k] * w[k] for owner in homeowner_list if owner != k))
    payment = sum((work_days[k][j] * w[j] for j in worker_list if j != k))
    m.addConstr(income == payment, name='fair_' + k)
m.setObjective(0, GRB.MINIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
    for worker in worker_list:
        print(f'{worker},{w[worker].X:.6f}')
else:
    print('No feasible solution found.')