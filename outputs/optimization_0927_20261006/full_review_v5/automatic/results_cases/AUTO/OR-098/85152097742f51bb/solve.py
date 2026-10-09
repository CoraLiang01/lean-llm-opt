import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',', dtype=str, keep_default_na=False)
all_columns = list(df.columns)
if all_columns[0] != 'Owner':
    raise ValueError("First column must be 'Owner'.")
worker_ids = all_columns[1:]
owner_ids = df['Owner'].tolist()
num_workers = len(worker_ids)
num_owners = len(owner_ids)
work_days_df = df.set_index('Owner')[worker_ids].apply(pd.to_numeric, errors='raise')
worker_total_days = work_days_df.sum(axis=0)
if not np.allclose(worker_total_days.values, 10):
    raise ValueError('Each worker must have exactly 10 total work days. Found: ' + str(worker_total_days.to_dict()))
m = gp.Model('mutual_wage_balance')
wage_vars = m.addVars(worker_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for j in worker_ids:
    income = gp.quicksum((work_days_df.at[i, j] * wage_vars[j] for i in owner_ids if i != j))
    expense = gp.quicksum((work_days_df.at[j, k] * wage_vars[k] for k in worker_ids if k != j))
    m.addConstr(income == expense, name=f'balance_{j}')
first_worker = worker_ids[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found:')
    for j in worker_ids:
        print(f'{j}: {wage_vars[j].X:.2f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')