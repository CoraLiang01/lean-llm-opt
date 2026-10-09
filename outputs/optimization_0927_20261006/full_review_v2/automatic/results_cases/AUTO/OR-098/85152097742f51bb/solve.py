import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
owners = df['Owner'].tolist()
missing_owners = set(worker_columns) - set(owners)
missing_workers = set(owners) - set(worker_columns)
if missing_owners or missing_workers:
    raise ValueError(f'Mismatch between worker columns and owner rows. Missing owners: {missing_owners}, Missing workers: {missing_workers}')
workers = worker_columns.copy()
work_days_df = df.set_index('Owner')[workers].applymap(int)
m = gp.Model('MutualAidWageBalance')
w_vars = m.addVars(workers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
for i in workers:
    income = gp.quicksum((work_days_df.loc[o, i] * w_vars[i] for o in owners if o != i))
    expense = gp.quicksum((work_days_df.loc[i, j] * w_vars[j] for j in workers if j != i))
    m.addConstr(income == expense, name=f'balance_{i}')
first_worker = workers[0]
m.addConstr(w_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print(f'Fixed wage: {first_worker} = {w_vars[first_worker].X:.2f} yuan/day')
    print('--- Daily wages for all workers ---')
    for j in workers:
        print(f'{j}: {w_vars[j].X:.6f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')