import gurobipy as gp
from gurobipy import GRB
workers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
time = {'A': [9, 4, 5, 7, 10, 6, 8, 7, 5, 8], 'B': [4, 6, 4, 5, 6, 7, 8, 4, 6, 7], 'C': [3, 5, 7, 2, 7, 6, 5, 8, 8, 9], 'D': [7, 6, 5, 3, 4, 3, 9, 8, 7, 5], 'E': [6, 4, 6, 7, 5, 9, 5, 6, 7, 8], 'F': [5, 5, 6, 8, 4, 5, 7, 7, 8, 5], 'G': [6, 3, 5, 5, 4, 7, 5, 5, 7, 9], 'H': [3, 8, 8, 6, 5, 4, 9, 7, 8, 9], 'I': [7, 7, 6, 8, 9, 3, 5, 7, 4, 3], 'J': [5, 6, 9, 5, 7, 4, 3, 7, 5, 4], 'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
for w in workers:
    if w not in time or len(time[w]) != len(tasks):
        raise ValueError(f'Missing or incomplete time data for worker {w}')
cost = {w: {t: time[w][i] for (i, t) in enumerate(tasks)} for w in workers}
m = gp.Model('worker_task_assignment')
x_vars = m.addVars(workers, tasks, vtype=GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][t] * x_vars[w, t] for w in workers for t in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[w, t] for w in workers)) == 1 for t in tasks), name='')
m.addConstrs((gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w] for w in workers), name='')
m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
m.addConstrs((gp.quicksum((x_vars[w, t] for t in tasks)) == y_vars[w] for w in workers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')