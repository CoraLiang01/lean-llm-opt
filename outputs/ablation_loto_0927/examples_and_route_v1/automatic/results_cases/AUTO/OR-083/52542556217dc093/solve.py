import gurobipy as gp
from gurobipy import GRB
workers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
time = {'A': [9, 4, 5, 7, 10, 6, 8, 7, 5, 8, 9, 8], 'B': [4, 6, 4, 5, 6, 7, 8, 4, 6, 7, 8, 5], 'C': [3, 5, 7, 2, 7, 6, 5, 8, 8, 9, 10, 6], 'D': [7, 6, 5, 3, 4, 3, 9, 8, 7, 5, 8, 9], 'E': [6, 4, 6, 7, 5, 9, 5, 6, 7, 8, 5, 4], 'F': [5, 5, 6, 8, 4, 5, 7, 7, 8, 5, 4, 7], 'G': [6, 3, 5, 5, 4, 7, 5, 5, 7, 9, 7, 8], 'H': [3, 8, 8, 6, 5, 4, 9, 7, 8, 9, 6, 4], 'I': [7, 7, 6, 8, 9, 3, 5, 7, 4, 3, 8, 7], 'J': [5, 6, 9, 5, 7, 4, 3, 7, 5, 4, 7, 9]}
if set(time.keys()) != set(workers):
    raise ValueError('Mismatch between workers and time matrix keys.')
for w in workers:
    if len(time[w]) != len(tasks):
        raise ValueError(f'Worker {w} does not have time data for all tasks.')
m = gp.Model('Assignment_10_of_12')
y = m.addVars(tasks, vtype=GRB.BINARY, name='')
z = m.addVars(workers, vtype=GRB.BINARY, name='')
x = m.addVars(workers, tasks, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((time[i][tasks.index(j)] * x[i, j] for i in workers for j in tasks)), GRB.MINIMIZE)
m.addConstr(gp.quicksum((y[j] for j in tasks)) == 10, name='select_10_tasks')
m.addConstr(gp.quicksum((z[i] for i in workers)) == 10, name='select_10_workers')
for j in tasks:
    m.addConstr(gp.quicksum((x[i, j] for i in workers)) == y[j], name=f'task_assign_{j}')
for i in workers:
    m.addConstr(gp.quicksum((x[i, j] for j in tasks)) == z[i], name=f'worker_assign_{i}')
for i in workers:
    for j in tasks:
        m.addConstr(x[i, j] <= y[j], name=f'x_leq_y_{i}_{j}')
        m.addConstr(x[i, j] <= z[i], name=f'x_leq_z_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')