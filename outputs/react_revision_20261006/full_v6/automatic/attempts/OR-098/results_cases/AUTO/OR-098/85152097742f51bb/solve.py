import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_cols = [col for col in df.columns if col != 'Owner']
owners = df['Owner'].tolist()
if not set(owners).issubset(set(worker_cols)):
    missing = set(owners) - set(worker_cols)
    raise ValueError(f'Owner(s) not found in worker columns: {missing}')
work_days_df = df[worker_cols].apply(pd.to_numeric, errors='raise')
owner_row_idx = {}
for (idx, owner) in enumerate(owners):
    if owner in owner_row_idx:
        raise ValueError(f"Duplicate owner row for worker '{owner}'")
    owner_row_idx[owner] = idx
for worker in worker_cols:
    total_days = work_days_df[worker].sum()
    if not np.isclose(total_days, 10):
        raise ValueError(f"Worker '{worker}' has total work days {total_days}, expected 10.")
workers = worker_cols
n_workers = len(workers)
m = gp.Model('mutual_wage_balance')
wage_vars = m.addVars(workers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
first_worker = workers[0]
m.addConstr(wage_vars[first_worker] == 60.0, name='fix_first_wage')
for j in workers:
    j_row = owner_row_idx[j]
    income_coeff = 0
    expense_expr = gp.LinExpr()
    income_coeff = work_days_df[j].sum() - work_days_df.loc[j_row, j]
    for k in workers:
        if k == j:
            continue
        days = work_days_df.loc[j_row, k]
        expense_expr.addTerms(days, wage_vars[k])
    m.addConstr(wage_vars[j] * income_coeff == expense_expr, name='bal_' + j)
m.setObjective(0.0, gp.GRB.MINIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()