import gurobipy as gp
from gurobipy import GRB
owners = ['Carpenter', 'Electrician', 'Painter', 'Plumber']
workers = ['Carpenter', 'Electrician', 'Painter', 'Plumber']
A = {'Carpenter': {'Carpenter': 2, 'Electrician': 3, 'Painter': 3, 'Plumber': 2}, 'Electrician': {'Carpenter': 3, 'Electrician': 2, 'Painter': 2, 'Plumber': 3}, 'Painter': {'Carpenter': 2, 'Electrician': 3, 'Painter': 2, 'Plumber': 3}, 'Plumber': {'Carpenter': 3, 'Electrician': 2, 'Painter': 3, 'Plumber': 2}}
N = len(owners)
if N != len(workers):
    raise ValueError('Number of owners and workers must match for this model.')
for i in owners:
    for j in workers:
        if j not in A[i]:
            raise ValueError(f'Missing A[{i}][{j}] in data.')
for j in workers:
    total_days = sum((A[i][j] for i in owners))
    if total_days != 10:
        raise ValueError(f'Worker {j} has total work days {total_days}, expected 10.')
m = gp.Model('mutual_payment')
p_vars = m.addVars(workers, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='')
m.addConstr(p_vars[workers[0]] == 60.0, name='wage_norm')
for k in range(N):
    owner_k = owners[k]
    worker_k = workers[k]
    lhs = gp.quicksum((A[owners[i]][worker_k] * p_vars[worker_k] for i in range(N) if i != k))
    rhs = gp.quicksum((A[owner_k][workers[j]] * p_vars[workers[j]] for j in range(N) if j != k))
    m.addConstr(lhs == rhs, name=f'balance_{worker_k}')
m.setObjective(0, GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')