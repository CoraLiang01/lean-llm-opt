import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    machines = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
    tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    cost = {'M1': {'T1': 90, 'T2': 76, 'T3': 75, 'T4': 70, 'T5': 50, 'T6': 74, 'T7': 80, 'T8': 85, 'T9': 65, 'T10': 48, 'T11': 50, 'T12': 55}, 'M2': {'T1': 35, 'T2': 85, 'T3': 55, 'T4': 65, 'T5': 48, 'T6': 101, 'T7': 70, 'T8': 83, 'T9': 78, 'T10': 64, 'T11': 60, 'T12': 59}, 'M3': {'T1': 125, 'T2': 95, 'T3': 90, 'T4': 105, 'T5': 59, 'T6': 120, 'T7': 36, 'T8': 73, 'T9': 80, 'T10': 40, 'T11': 55, 'T12': 60}, 'M4': {'T1': 45, 'T2': 110, 'T3': 95, 'T4': 115, 'T5': 104, 'T6': 83, 'T7': 37, 'T8': 71, 'T9': 67, 'T10': 53, 'T11': 60, 'T12': 65}, 'M5': {'T1': 60, 'T2': 105, 'T3': 80, 'T4': 75, 'T5': 59, 'T6': 62, 'T7': 77, 'T8': 90, 'T9': 80, 'T10': 55, 'T11': 80, 'T12': 80}, 'M6': {'T1': 45, 'T2': 65, 'T3': 110, 'T4': 95, 'T5': 47, 'T6': 31, 'T7': 81, 'T8': 74, 'T9': 70, 'T10': 59, 'T11': 63, 'T12': 65}, 'M7': {'T1': 60, 'T2': 45, 'T3': 80, 'T4': 75, 'T5': 59, 'T6': 62, 'T7': 77, 'T8': 90, 'T9': 80, 'T10': 55, 'T11': 80, 'T12': 80}, 'M8': {'T1': 45, 'T2': 65, 'T3': 110, 'T4': 95, 'T5': 47, 'T6': 31, 'T7': 81, 'T8': 74, 'T9': 70, 'T10': 59, 'T11': 63, 'T12': 65}, 'M9': {'T1': 60, 'T2': 105, 'T3': 80, 'T4': 75, 'T5': 59, 'T6': 62, 'T7': 77, 'T8': 90, 'T9': 80, 'T10': 55, 'T11': 80, 'T12': 80}, 'M10': {'T1': 45, 'T2': 65, 'T3': 110, 'T4': 95, 'T5': 47, 'T6': 31, 'T7': 81, 'T8': 74, 'T9': 70, 'T10': 59, 'T11': 63, 'T12': 65}, 'M11': {'T1': 60, 'T2': 105, 'T3': 80, 'T4': 75, 'T5': 59, 'T6': 62, 'T7': 77, 'T8': 90, 'T9': 80, 'T10': 55, 'T11': 80, 'T12': 80}, 'M12': {'T1': 45, 'T2': 65, 'T3': 110, 'T4': 95, 'T5': 47, 'T6': 31, 'T7': 81, 'T8': 74, 'T9': 70, 'T10': 59, 'T11': 63, 'T12': 65}}
    for i in machines:
        if i not in cost:
            raise ValueError(f'Missing cost data for machine {i}')
        for j in tasks:
            if j not in cost[i]:
                raise ValueError(f'Missing cost data for machine {i}, task {j}')
    m = gp.Model('FactoryAssignment')
    x = m.addVars(machines, tasks, vtype=GRB.BINARY, name='x')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in machines for j in tasks)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for j in tasks)) == 1 for i in machines), name='machine_assign')
    m.addConstrs((gp.quicksum((x[i, j] for i in machines)) == 1 for j in tasks), name='task_assign')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()