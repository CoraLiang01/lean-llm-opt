import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    machines = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
    tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    cost = {'M1': {'T1': 90, 'T2': 76, 'T3': 75, 'T4': 70, 'T5': 50, 'T6': 74, 'T7': 80, 'T8': 85, 'T9': 65, 'T10': 48, 'T11': 50, 'T12': 55}, 'M2': {'T1': 35, 'T2': 85, 'T3': 55, 'T4': 65, 'T5': 48, 'T6': 101, 'T7': 70, 'T8': 83, 'T9': 78, 'T10': 60, 'T11': 59, 'T12': 60}, 'M3': {'T1': 125, 'T2': 95, 'T3': 90, 'T4': 105, 'T5': 59, 'T6': 120, 'T7': 95, 'T8': 105, 'T9': 80, 'T10': 80, 'T11': 80, 'T12': 90}, 'M4': {'T1': 45, 'T2': 110, 'T3': 95, 'T4': 115, 'T5': 104, 'T6': 83, 'T7': 97, 'T8': 99, 'T9': 70, 'T10': 60, 'T11': 59, 'T12': 63}, 'M5': {'T1': 60, 'T2': 105, 'T3': 80, 'T4': 75, 'T5': 59, 'T6': 62, 'T7': 77, 'T8': 90, 'T9': 80, 'T10': 55, 'T11': 60, 'T12': 65}, 'M6': {'T1': 45, 'T2': 65, 'T3': 110, 'T4': 95, 'T5': 47, 'T6': 31, 'T7': 81, 'T8': 99, 'T9': 60, 'T10': 50, 'T11': 57, 'T12': 60}, 'M7': {'T1': 60, 'T2': 70, 'T3': 60, 'T4': 80, 'T5': 50, 'T6': 65, 'T7': 80, 'T8': 80, 'T9': 60, 'T10': 50, 'T11': 60, 'T12': 65}, 'M8': {'T1': 55, 'T2': 65, 'T3': 60, 'T4': 80, 'T5': 48, 'T6': 50, 'T7': 65, 'T8': 70, 'T9': 60, 'T10': 50, 'T11': 60, 'T12': 65}, 'M9': {'T1': 80, 'T2': 80, 'T3': 80, 'T4': 80, 'T5': 80, 'T6': 80, 'T7': 80, 'T8': 80, 'T9': 80, 'T10': 80, 'T11': 80, 'T12': 80}, 'M10': {'T1': 90, 'T2': 90, 'T3': 90, 'T4': 90, 'T5': 90, 'T6': 90, 'T7': 90, 'T8': 90, 'T9': 90, 'T10': 90, 'T11': 90, 'T12': 90}, 'M11': {'T1': 50, 'T2': 50, 'T3': 50, 'T4': 50, 'T5': 50, 'T6': 50, 'T7': 50, 'T8': 50, 'T9': 50, 'T10': 50, 'T11': 50, 'T12': 50}, 'M12': {'T1': 60, 'T2': 60, 'T3': 60, 'T4': 60, 'T5': 60, 'T6': 60, 'T7': 60, 'T8': 60, 'T9': 60, 'T10': 60, 'T11': 60, 'T12': 60}}
    for i in machines:
        if i not in cost or set(cost[i].keys()) != set(tasks):
            raise ValueError(f'Cost data missing for machine {i} or its tasks.')
    m = gp.Model('FactoryAssignment')
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
    return m
m = solve_problem()