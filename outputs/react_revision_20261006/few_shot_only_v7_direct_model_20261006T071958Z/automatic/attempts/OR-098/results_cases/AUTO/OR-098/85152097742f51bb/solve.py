import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_cols = [col for col in df.columns if col != 'Owner']
num_workers = len(worker_cols)
owners = df['Owner'].tolist()
if set(owners) != set(worker_cols):
    raise ValueError('Mismatch between Owner values and worker columns. Each Owner must correspond to a worker column.')
days_df = df[worker_cols].apply(pd.to_numeric, errors='raise')
owner_to_row = {owner: idx for (idx, owner) in enumerate(owners)}
for j in worker_cols:
    total_days = days_df[j].sum()
    if not np.isclose(total_days, 10.0, atol=1e-06):
        raise ValueError(f'Worker {j} has total work days {total_days}, expected 10.')
m = gp.Model('mutual_payment_wages')
wage_vars = m.addVars(worker_cols, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = worker_cols[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='norm')
for k in worker_cols:
    income = gp.LinExpr()
    for (i, owner) in enumerate(owners):
        if owner != k:
            income += days_df.at[i, k] * wage_vars[k]
    k_row = owner_to_row[k]
    expenditure = gp.LinExpr()
    for j in worker_cols:
        if j != k:
            expenditure += days_df.at[k_row, j] * wage_vars[j]
    m.addConstr(income == expenditure, name='fair_' + k)
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for j in worker_cols:
        print(f'{wage_vars[j].VarName} {wage_vars[j].X}')
else:
    print(f'Solver status: {m.status}')