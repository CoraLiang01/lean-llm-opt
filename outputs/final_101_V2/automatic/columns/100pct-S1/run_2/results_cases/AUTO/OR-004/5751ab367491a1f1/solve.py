import gurobipy as gp
from gurobipy import GRB
machines = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
cost = {'M1': {'T1': 90, 'T2': 76, 'T3': 75, 'T4': 70, 'T5': 50, 'T6': 74, 'T7': 80, 'T8': 74, 'T9': 65, 'T10': 48, 'T11': 78, 'T12': 50}, 'M2': {'T1': 35, 'T2': 85, 'T3': 55, 'T4': 65, 'T5': 48, 'T6': 101, 'T7': 70, 'T8': 83, 'T9': 78, 'T10': 64, 'T11': 85, 'T12': 60}, 'M3': {'T1': 125, 'T2': 95, 'T3': 90, 'T4': 105, 'T5': 59, 'T6': 120, 'T7': 36, 'T8': 73, 'T9': 89, 'T10': 59, 'T11': 77, 'T12': 50}, 'M4': {'T1': 45, 'T2': 110, 'T3': 95, 'T4': 115, 'T5': 104, 'T6': 83, 'T7': 37, 'T8': 63, 'T9': 64, 'T10': 61, 'T11': 92, 'T12': 77}, 'M5': {'T1': 60, 'T2': 105, 'T3': 80, 'T4': 75, 'T5': 59, 'T6': 62, 'T7': 93, 'T8': 88, 'T9': 49, 'T10': 68, 'T11': 74, 'T12': 60}, 'M6': {'T1': 45, 'T2': 65, 'T3': 110, 'T4': 95, 'T5': 47, 'T6': 31, 'T7': 81, 'T8': 34, 'T9': 53, 'T10': 46, 'T11': 62, 'T12': 69}, 'M7': {'T1': 38, 'T2': 51, 'T3': 107, 'T4': 41, 'T5': 69, 'T6': 99, 'T7': 115, 'T8': 48, 'T9': 48, 'T10': 52, 'T11': 68, 'T12': 60}, 'M8': {'T1': 47, 'T2': 85, 'T3': 57, 'T4': 71, 'T5': 92, 'T6': 77, 'T7': 109, 'T8': 36, 'T9': 47, 'T10': 99, 'T11': 65, 'T12': 76}, 'M9': {'T1': 39, 'T2': 63, 'T3': 97, 'T4': 49, 'T5': 121, 'T6': 83, 'T7': 75, 'T8': 56, 'T9': 44, 'T10': 61, 'T11': 54, 'T12': 60}, 'M10': {'T1': 47, 'T2': 101, 'T3': 71, 'T4': 60, 'T5': 88, 'T6': 57, 'T7': 98, 'T8': 68, 'T9': 60, 'T10': 61, 'T11': 50, 'T12': 60}, 'M11': {'T1': 39, 'T2': 63, 'T3': 97, 'T4': 49, 'T5': 121, 'T6': 83, 'T7': 75, 'T8': 56, 'T9': 44, 'T10': 61, 'T11': 54, 'T12': 60}, 'M12': {'T1': 50, 'T2': 60, 'T3': 57, 'T4': 60, 'T5': 50, 'T6': 60, 'T7': 60, 'T8': 50, 'T9': 60, 'T10': 50, 'T11': 60, 'T12': 50}}
if set(cost.keys()) != set(machines):
    raise ValueError('Cost matrix missing machine keys')
for m in machines:
    if set(cost[m].keys()) != set(tasks):
        raise ValueError(f'Cost matrix missing task keys for machine {m}')
m = gp.Model('Factory_Assignment')
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