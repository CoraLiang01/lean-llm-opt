import gurobipy as gp
from gurobipy import GRB
models = [f'HiFi-{k}' for k in range(1, 102)]
a_1 = [6, 4, 6] + [7] * 97 + [9]
a_2 = [5, 5, 5] + [4] * 97 + [3]
a_3 = [4, 6, 5] + [5] * 97 + [6]
if not (len(a_1) == len(models) and len(a_2) == len(models) and (len(a_3) == len(models))):
    raise ValueError('Processing time lists must have 101 entries each.')
processing_times = {1: dict(zip(models, a_1)), 2: dict(zip(models, a_2)), 3: dict(zip(models, a_3))}
capacities = {1: 1296, 2: 1238.4, 3: 1267.2}
for i in [1, 2, 3]:
    if set(processing_times[i].keys()) != set(models):
        raise ValueError(f'Missing processing times for workstation {i}')
m = gp.Model('Radio_Idle_Minimization')
x = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
total_proc = gp.quicksum((processing_times[i][k] * x[k] for i in [1, 2, 3] for k in models))
total_capacity = sum((capacities[i] for i in [1, 2, 3]))
m.setObjective(total_capacity - total_proc, GRB.MINIMIZE)
for i in [1, 2, 3]:
    m.addConstr(gp.quicksum((processing_times[i][k] * x[k] for k in models)) <= capacities[i], name=f'cap{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')