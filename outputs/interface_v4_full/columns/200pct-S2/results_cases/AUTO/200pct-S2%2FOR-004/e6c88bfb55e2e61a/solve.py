import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    machines = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
    tasks = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    cost = {'M1': {'T1': 8, 'T2': 15, 'T3': 13, 'T4': 17, 'T5': 20, 'T6': 12, 'T7': 14, 'T8': 16, 'T9': 18, 'T10': 11, 'T11': 19, 'T12': 21}, 'M2': {'T1': 14, 'T2': 9, 'T3': 16, 'T4': 12, 'T5': 11, 'T6': 18, 'T7': 17, 'T8': 13, 'T9': 15, 'T10': 20, 'T11': 21, 'T12': 19}, 'M3': {'T1': 13, 'T2': 17, 'T3': 8, 'T4': 15, 'T5': 19, 'T6': 14, 'T7': 12, 'T8': 20, 'T9': 16, 'T10': 18, 'T11': 21, 'T12': 11}, 'M4': {'T1': 15, 'T2': 12, 'T3': 14, 'T4': 8, 'T5': 13, 'T6': 17, 'T7': 19, 'T8': 11, 'T9': 21, 'T10': 16, 'T11': 18, 'T12': 20}, 'M5': {'T1': 17, 'T2': 13, 'T3': 15, 'T4': 19, 'T5': 8, 'T6': 21, 'T7': 18, 'T8': 12, 'T9': 14, 'T10': 20, 'T11': 16, 'T12': 11}, 'M6': {'T1': 12, 'T2': 18, 'T3': 17, 'T4': 14, 'T5': 21, 'T6': 8, 'T7': 13, 'T8': 19, 'T9': 11, 'T10': 15, 'T11': 20, 'T12': 16}, 'M7': {'T1': 16, 'T2': 11, 'T3': 19, 'T4': 13, 'T5': 18, 'T6': 15, 'T7': 8, 'T8': 21, 'T9': 20, 'T10': 12, 'T11': 14, 'T12': 17}, 'M8': {'T1': 11, 'T2': 20, 'T3': 21, 'T4': 16, 'T5': 12, 'T6': 13, 'T7': 15, 'T8': 8, 'T9': 17, 'T10': 14, 'T11': 18, 'T12': 19}, 'M9': {'T1': 19, 'T2': 14, 'T3': 12, 'T4': 21, 'T5': 16, 'T6': 20, 'T7': 11, 'T8': 18, 'T9': 8, 'T10': 13, 'T11': 15, 'T12': 17}, 'M10': {'T1': 21, 'T2': 16, 'T3': 18, 'T4': 20, 'T5': 14, 'T6': 19, 'T7': 16, 'T8': 17, 'T9': 13, 'T10': 8, 'T11': 12, 'T12': 15}, 'M11': {'T1': 18, 'T2': 21, 'T3': 20, 'T4': 11, 'T5': 15, 'T6': 16, 'T7': 17, 'T8': 14, 'T9': 12, 'T10': 19, 'T11': 8, 'T12': 13}, 'M12': {'T1': 20, 'T2': 19, 'T3': 11, 'T4': 18, 'T5': 17, 'T6': 15, 'T7': 21, 'T8': 10, 'T9': 19, 'T10': 12, 'T11': 13, 'T12': 8}}
    if set(cost.keys()) != set(machines):
        raise ValueError('Cost matrix row keys do not match machines.')
    for m in machines:
        if set(cost[m].keys()) != set(tasks):
            raise ValueError(f'Cost matrix columns for {m} do not match tasks.')
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
    return m
m = solve_problem()