import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col != 'Owner']
owner_ids = df['Owner'].tolist()
if set(owner_ids) != set(worker_columns):
    raise ValueError('Mismatch between Owner rows and worker columns.')
workers = worker_columns
D = df.set_index('Owner')[workers].applymap(lambda x: int(x.strip()))
for j in workers:
    total_days = D[j].sum()
    if total_days != 10:
        raise ValueError(f'Worker {j} has total work days {total_days}, expected 10.')
m = gp.Model('Mutual_Payment_Wages')
wage_vars = m.addVars(workers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for k in workers:
    payment_out = gp.quicksum((D.loc[k, j] * wage_vars[j] for j in workers if j != k))
    payment_in = wage_vars[k] * sum((D.loc[i, k] for i in workers if i != k))
    m.addConstr(payment_out == payment_in, name=f'balance_{k}')
first_worker = workers[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found:')
    for j in workers:
        print(f'{j}: {wage_vars[j].X:.2f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')