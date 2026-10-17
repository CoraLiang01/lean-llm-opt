import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_columns)
num_owners = df.shape[0]
owner_ids = df['Owner'].tolist()
d_matrix = df[worker_columns].astype(int).to_numpy()
worker_idx_to_name = {j: worker_columns[j] for j in range(num_workers)}
worker_name_to_idx = {worker_columns[j]: j for j in range(num_workers)}
m = gp.Model('Mutual_Wage_Balance')
wage_vars = m.addVars(worker_columns, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for j, worker_j in enumerate(worker_columns):
    income = sum((d_matrix[i, j] * wage_vars[worker_j] for i in range(num_owners)))
    expense = gp.quicksum((d_matrix[j, k] * wage_vars[worker_idx_to_name[k]] for k in range(num_workers)))
    m.addConstr(income - expense == 0, name=f'balance_{worker_j}')
first_worker = worker_columns[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found:')
    for worker in worker_columns:
        print(f'{worker}: {wage_vars[worker].X:.6f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')