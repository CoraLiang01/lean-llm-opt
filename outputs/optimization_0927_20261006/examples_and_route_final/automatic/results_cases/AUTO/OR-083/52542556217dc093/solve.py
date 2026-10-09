import gurobipy as gp
from gurobipy import GRB
workers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
tasks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
time = {'A': {'1': 9, '2': 4, '3': 5, '4': 7, '5': 10, '6': 6, '7': 8, '8': 7, '9': 5, '10': 8}, 'B': {'1': 4, '2': 6, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 4, '9': 6, '10': 7}, 'C': {'1': 3, '2': 5, '3': 7, '4': 2, '5': 7, '6': 6, '7': 5, '8': 8, '9': 8, '10': 9}, 'D': {'1': 7, '2': 6, '3': 5, '4': 3, '5': 4, '6': 3, '7': 9, '8': 8, '9': 7, '10': 5}, 'E': {'1': 6, '2': 4, '3': 6, '4': 7, '5': 5, '6': 9, '7': 5, '8': 6, '9': 7, '10': 8}, 'F': {'1': 5, '2': 5, '3': 6, '4': 8, '5': 4, '6': 5, '7': 7, '8': 7, '9': 8, '10': 5}, 'G': {'1': 6, '2': 3, '3': 5, '4': 5, '5': 4, '6': 7, '7': 5, '8': 5, '9': 7, '10': 9}, 'H': {'1': 3, '2': 8, '3': 8, '4': 6, '5': 5, '6': 4, '7': 9, '8': 7, '9': 8, '10': 9}, 'I': {'1': 7, '2': 7, '3': 6, '4': 8, '5': 9, '6': 3, '7': 5, '8': 7, '9': 4, '10': 3}, 'J': {'1': 5, '2': 6, '3': 9, '4': 5, '5': 7, '6': 4, '7': 3, '8': 7, '9': 5, '10': 4}}
for w in workers:
    if w not in time:
        raise ValueError(f'Missing time data for worker {w}')
    for t in tasks:
        if t not in time[w]:
            raise ValueError(f'Missing time data for worker {w}, task {t}')
m = gp.Model('worker_task_assignment')
x_vars = m.addVars(workers, tasks, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((time[w][t] * x_vars[w, t] for w in workers for t in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[w, t] for w in workers)) == 1 for t in tasks), name='')
m.addConstrs((gp.quicksum((x_vars[w, t] for t in tasks)) <= 1 for w in workers), name='')
m.addConstr(gp.quicksum((x_vars[w, t] for w in workers for t in tasks)) == 10, name='total_assign')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')