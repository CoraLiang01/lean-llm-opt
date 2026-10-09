import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_cols = [col for col in df.columns if col != 'Owner']
homeowners = df['Owner'].tolist()
if set(worker_cols) != set(homeowners):
    raise ValueError("Mismatch between worker columns and homeowner names in 'Owner' column.")
if len(worker_cols) != len(homeowners):
    raise ValueError('Number of workers and homeowners does not match.')
worker_list = worker_cols
num_workers = len(worker_list)
D_df = df.set_index('Owner')[worker_list].applymap(lambda x: float(x.strip()) if x.strip() != '' else 0.0)
m = gp.Model('MutualAidWageBalance')
wage_vars = m.addVars(worker_list, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
for i in worker_list:
    income = gp.quicksum((D_df.loc[j, i] * wage_vars[i] for j in worker_list))
    payment = gp.quicksum((D_df.loc[i, j] * wage_vars[j] for j in worker_list))
    m.addConstr(income == payment, name=f'balance_{i}')
first_worker = worker_list[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='wage_normalization')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print('Daily wages (yuan):')
    for w in worker_list:
        print(f'{w}: {wage_vars[w].X:.6f}')
else:
    print(f'No feasible solution found. Status: {m.status}')