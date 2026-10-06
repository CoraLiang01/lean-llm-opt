import gurobipy as gp
from gurobipy import GRB
machines = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
cost = {'M1': {'T1': 90, 'T2': 76, 'T3': 75, 'T4': 70, 'T5': 50, 'T6': 74, 'T7': 80, 'T8': 85, 'T9': 65, 'T10': 48, 'T11': 50, 'T12': 55}, 'M2': {'T1': 35, 'T2': 85, 'T3': 55, 'T4': 65, 'T5': 48, 'T6': 101, 'T7': 70, 'T8': 83, 'T9': 78, 'T10': 60, 'T11': 59, 'T12': 55}, 'M3': {'T1': 125, 'T2': 95, 'T3': 90, 'T4': 105, 'T5': 59, 'T6': 120, 'T7': 36, 'T8': 73, 'T9': 62, 'T10': 80, 'T11': 40, 'T12': 70}, 'M4': {'T1': 45, 'T2': 110, 'T3': 95, 'T4': 115, 'T5': 104, 'T6': 83, 'T7': 37, 'T8': 63, 'T9': 64, 'T10': 61, 'T11': 60, 'T12': 59}, 'M5': {'T1': 60, 'T2': 105, 'T3': 80, 'T4': 75, 'T5': 59, 'T6': 62, 'T7': 93, 'T8': 88, 'T9': 56, 'T10': 49, 'T11': 50, 'T12': 60}, 'M6': {'T1': 45, 'T2': 65, 'T3': 110, 'T4': 95, 'T5': 47, 'T6': 31, 'T7': 81, 'T8': 34, 'T9': 53, 'T10': 46, 'T11': 50, 'T12': 65}, 'M7': {'T1': 38, 'T2': 51, 'T3': 60, 'T4': 107, 'T5': 95, 'T6': 99, 'T7': 68, 'T8': 80, 'T9': 60, 'T10': 59, 'T11': 55, 'T12': 53}, 'M8': {'T1': 47, 'T2': 85, 'T3': 57, 'T4': 71, 'T5': 92, 'T6': 77, 'T7': 109, 'T8': 36, 'T9': 63, 'T10': 65, 'T11': 51, 'T12': 60}, 'M9': {'T1': 39, 'T2': 63, 'T3': 97, 'T4': 49, 'T5': 121, 'T6': 83, 'T7': 75, 'T8': 59, 'T9': 62, 'T10': 57, 'T11': 55, 'T12': 61}, 'M10': {'T1': 40, 'T2': 73, 'T3': 75, 'T4': 63, 'T5': 64, 'T6': 78, 'T7': 50, 'T8': 60, 'T9': 80, 'T10': 57, 'T11': 60, 'T12': 74}, 'M11': {'T1': 60, 'T2': 60, 'T3': 80, 'T4': 75, 'T5': 75, 'T6': 80, 'T7': 55, 'T8': 80, 'T9': 80, 'T10': 60, 'T11': 60, 'T12': 60}, 'M12': {'T1': 50, 'T2': 65, 'T3': 60, 'T4': 59, 'T5': 55, 'T6': 65, 'T7': 60, 'T8': 80, 'T9': 80, 'T10': 60, 'T11': 58, 'T12': 60}}
if set(cost.keys()) != set(machines):
    raise ValueError('Cost data missing for some machines.')
for i in machines:
    if set(cost[i].keys()) != set(tasks):
        raise ValueError(f'Cost data missing for some tasks for machine {i}.')
m = gp.Model('Assignment_12x12')
x = m.addVars(machines, tasks, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in machines for j in tasks)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in tasks)) == 1 for i in machines), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in machines)) == 1 for j in tasks), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')