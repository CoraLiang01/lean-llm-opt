import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv'
    df = pd.read_csv(path, sep=',')
    worker_cols = [col for col in df.columns if col != 'Owner']
    workers = list(worker_cols)
    n_workers = len(workers)
    homeowners = df['Owner'].astype(str).tolist()
    if len(homeowners) != n_workers:
        raise ValueError(f'Number of homeowners ({len(homeowners)}) does not match number of workers ({n_workers})')
    worker_to_row = {workers[i]: i for i in range(n_workers)}
    row_to_worker = {i: workers[i] for i in range(n_workers)}
    days = df[worker_cols].to_numpy(dtype=float)
    total_days_by_worker = days.sum(axis=0)
    if not np.allclose(total_days_by_worker, 10.0):
        raise ValueError('Each worker must contribute exactly 10 work days. Found: ' + str(dict(zip(workers, total_days_by_worker))))
    m = gp.Model('mutual_wage_payment')
    m.Params.OutputFlag = 0
    w = {}
    for j in workers:
        if j == workers[0]:
            w[j] = m.addVar(lb=60.0, ub=60.0, vtype=GRB.CONTINUOUS, name=f'w_{j}')
        else:
            w[j] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f'w_{j}')
    m.update()
    for i in range(n_workers):
        wi = row_to_worker[i]
        lhs = float(days[:, i].sum()) * w[wi]
        rhs = gp.LinExpr()
        for j in range(n_workers):
            wj = row_to_worker[j]
            rhs.addTerms(float(days[i, j]), w[wj])
        m.addConstr(lhs == rhs, name=f'fairness_{wi}')
    m.setObjective(0.0, GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()