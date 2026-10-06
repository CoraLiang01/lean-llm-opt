import gurobipy as gp
import pandas as pd
import numpy as np
work_days_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
df = pd.read_csv(work_days_path, sep=',')
if 'Owner' not in df.columns:
    raise ValueError("Missing required column 'Owner' in work_days.csv")
worker_cols = [col for col in df.columns if col != 'Owner']
owners = df['Owner'].astype(str).tolist()
if set(owners) != set(worker_cols):
    raise ValueError('Mismatch between owners and workers. Each worker must also be an owner.')
workers = worker_cols
owners = workers
num_workers = len(workers)
work_days = df.set_index('Owner').loc[owners, workers].astype(float).values
total_days_by_worker = work_days.sum(axis=0)
if not np.allclose(total_days_by_worker, 10.0, atol=1e-06):
    raise ValueError('Each worker must contribute exactly 10 work days in total.')

def solve_problem(workers, owners, work_days):
    m = gp.Model('mutual_payment_balance')
    m.Params.MIPGap = 0.0001
    w = m.addVars(workers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    first_worker = workers[0]
    m.addConstr(w[first_worker] == 60.0, name='fix_first_wage')
    for (idx_k, k) in enumerate(workers):
        income = sum((work_days[i, idx_k] for i in range(num_workers))) * w[k]
        expenditure = gp.quicksum((work_days[idx_k, idx_j] * w[workers[idx_j]] for idx_j in range(num_workers)))
        m.addConstr(income - expenditure == 0, name='balance_' + k)
    m.setObjective(0.0, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem(workers, owners, work_days)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X:.6f}')
else:
    print(f'Solver status: {m.status}')