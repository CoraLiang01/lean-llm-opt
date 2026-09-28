import gurobipy as gp
from gurobipy import GRB
workers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
time = {'A': [9, 4, 3, 7, 6, 5, 6, 3, 7, 5], 'B': [4, 6, 5, 6, 4, 5, 3, 8, 7, 6], 'C': [5, 4, 7, 5, 6, 6, 5, 8, 6, 9], 'D': [7, 5, 2, 3, 7, 8, 5, 6, 8, 5], 'E': [10, 6, 7, 4, 5, 4, 4, 5, 9, 7], 'F': [6, 7, 6, 3, 9, 5, 7, 4, 3, 4], 'G': [8, 8, 5, 9, 5, 7, 5, 9, 5, 3], 'H': [7, 4, 8, 8, 6, 7, 5, 7, 7, 7], 'I': [5, 6, 8, 7, 7, 8, 7, 8, 4, 5], 'J': [8, 7, 9, 5, 8, 5, 9, 9, 3, 4], 'K': [9, 8, 10, 8, 5, 4, 7, 6, 8, 7], 'L': [8, 5, 6, 9, 4, 7, 8, 4, 7, 9]}
if set(time.keys()) != set(workers):
    raise ValueError('Mismatch between workers and time keys')
for w in workers:
    if len(time[w]) != len(tasks):
        raise ValueError(f'Worker {w} does not have time data for all tasks')
c = {w: {t: time[w][i] for i, t in enumerate(tasks)} for w in workers}
m = gp.Model('worker_task_assignment')
x = m.addVars(workers, tasks, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[w][t] * x[w, t] for w in workers for t in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[w, t] for w in workers)) == 1 for t in tasks), name='')
m.addConstrs((gp.quicksum((x[w, t] for t in tasks)) <= 1 for w in workers), name='')
m.addConstr(gp.quicksum((x[w, t] for w in workers for t in tasks)) == 10, name='total_assignments')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')