import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_columns = [col for col in df.columns if col.strip() != '']
if len(worker_columns) == 0:
    raise ValueError('No worker columns found in the CSV.')
workers = worker_columns
num_workers = len(workers)
work_days = pd.DataFrame({col: pd.to_numeric(df[col].str.strip(), errors='raise') for col in workers})
owners = work_days.index.tolist()
if len(owners) != num_workers:
    raise ValueError('Number of owners (rows) does not match number of workers (columns).')
owner_names = df.index.tolist()
if len(owner_names) != num_workers:
    owner_names = [str(i) for i in range(num_workers)]
for j in workers:
    total_days = work_days[j].sum()
    if not np.isclose(total_days, 10.0, atol=1e-06):
        raise ValueError(f"Worker '{j}' does not have exactly 10 total work days (has {total_days}).")
m = gp.Model('mutual_payment_wages')
wage_vars = m.addVars(workers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = workers[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fixed_wage')
for k_idx, k in enumerate(workers):
    income = gp.quicksum((work_days.iloc[i, workers.index(k)] * wage_vars[k] for i in range(num_workers) if i != k_idx))
    expenditure = gp.quicksum((work_days.iloc[k_idx, workers.index(j)] * wage_vars[j] for j in workers if j != k))
    m.addConstr(income == expenditure, name=f'balance_{k}')
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL or m.status == gp.GRB.SUBOPTIMAL:
    print('Feasible daily wages found:')
    for j in workers:
        print(f'  {j}: {wage_vars[j].X:.2f} yuan/day')
else:
    print(f'No feasible solution found. Status: {m.status}')