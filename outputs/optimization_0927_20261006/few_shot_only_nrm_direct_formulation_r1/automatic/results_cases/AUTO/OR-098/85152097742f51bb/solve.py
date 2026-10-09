import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_cols = [col for col in df.columns if col != 'Owner']
workers = list(worker_cols)
owners = df['Owner'].tolist()
if not set(owners).issubset(set(workers)):
    missing = set(owners) - set(workers)
    raise ValueError(f'Owner(s) not found among workers: {missing}')
D = df.set_index('Owner')[worker_cols].applymap(lambda x: float(x.strip()) if x.strip() != '' else 0.0)
m = gp.Model('MutualAidWageFairness')
w_vars = m.addVars(workers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for k in workers:
    income = gp.quicksum((D.loc[i, k] * w_vars[k] for i in owners))
    expenditure = gp.quicksum((D.loc[k, j] * w_vars[j] for j in workers))
    m.addConstr(income == expenditure, name=f'fair_{k}')
first_worker = workers[0]
m.addConstr(w_vars[first_worker] == 60.0, name='fixed_wage')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible wage solution found.')
    print(f'Fixed wage: {first_worker} = {w_vars[first_worker].X:.2f} yuan/day')
    print('--- Daily wages for all workers ---')
    for j in workers:
        print(f'{j}: {w_vars[j].X:.2f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')