import gurobipy as gp
from gurobipy import GRB
workers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
tasks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
time = {'1': {'A': 9, 'B': 4, 'C': 3, 'D': 7, 'E': 6, 'F': 5, 'G': 6, 'H': 3, 'I': 7, 'J': 5}, '2': {'A': 4, 'B': 6, 'C': 5, 'D': 6, 'E': 4, 'F': 5, 'G': 3, 'H': 8, 'I': 7, 'J': 6}, '3': {'A': 5, 'B': 4, 'C': 7, 'D': 5, 'E': 6, 'F': 6, 'G': 5, 'H': 8, 'I': 6, 'J': 9}, '4': {'A': 7, 'B': 5, 'C': 2, 'D': 3, 'E': 7, 'F': 8, 'G': 5, 'H': 6, 'I': 8, 'J': 5}, '5': {'A': 10, 'B': 6, 'C': 7, 'D': 4, 'E': 5, 'F': 4, 'G': 4, 'H': 5, 'I': 9, 'J': 7}, '6': {'A': 6, 'B': 7, 'C': 6, 'D': 3, 'E': 9, 'F': 5, 'G': 7, 'H': 4, 'I': 3, 'J': 4}, '7': {'A': 8, 'B': 8, 'C': 5, 'D': 9, 'E': 5, 'F': 7, 'G': 5, 'H': 9, 'I': 5, 'J': 3}, '8': {'A': 7, 'B': 4, 'C': 8, 'D': 8, 'E': 6, 'F': 7, 'G': 5, 'H': 7, 'I': 7, 'J': 7}, '9': {'A': 5, 'B': 6, 'C': 8, 'D': 7, 'E': 7, 'F': 8, 'G': 7, 'H': 8, 'I': 4, 'J': 5}, '10': {'A': 8, 'B': 7, 'C': 9, 'D': 5, 'E': 8, 'F': 5, 'G': 9, 'H': 9, 'I': 3, 'J': 4}, '11': {'A': 9, 'B': 8, 'C': 10, 'D': 8, 'E': 5, 'F': 4, 'G': 7, 'H': 6, 'I': 8, 'J': 7}, '12': {'A': 8, 'B': 5, 'C': 6, 'D': 9, 'E': 4, 'F': 7, 'G': 8, 'H': 4, 'I': 7, 'J': 9}}
for w in workers:
    if w not in time:
        raise ValueError(f'Missing time data for worker {w}')
    for t in tasks:
        if t not in time[w]:
            raise ValueError(f'Missing time data for worker {w}, task {t}')
m = gp.Model('worker_task_assignment')
x = m.addVars(workers, tasks, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((time[w][t] * x[w, t] for w in workers for t in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[w, t] for w in workers)) == 1 for t in tasks), name='')
m.addConstrs((gp.quicksum((x[w, t] for t in tasks)) <= 1 for w in workers), name='')
m.addConstr(gp.quicksum((x[w, t] for w in workers for t in tasks)) == 10, name='total_assignments')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')