import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    facilities = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6']
    neighborhoods = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10']
    demand = {'N1': 28, 'N2': 32, 'N3': 44, 'N4': 36, 'N5': 52, 'N6': 41, 'N7': 25, 'N8': 48, 'N9': 55, 'N10': 30}
    distance = {'F1': {'N1': 2, 'N2': 3, 'N3': 5, 'N4': 9, 'N5': 10, 'N6': 11, 'N7': 13, 'N8': 14, 'N9': 15, 'N10': 12}, 'F2': {'N1': 4, 'N2': 2, 'N3': 3, 'N4': 8, 'N5': 9, 'N6': 10, 'N7': 12, 'N8': 13, 'N9': 14, 'N10': 11}, 'F3': {'N1': 9, 'N2': 8, 'N3': 4, 'N4': 2, 'N5': 3, 'N6': 5, 'N7': 9, 'N8': 10, 'N9': 11, 'N10': 7}, 'F4': {'N1': 10, 'N2': 9, 'N3': 6, 'N4': 3, 'N5': 2, 'N6': 3, 'N7': 8, 'N8': 9, 'N9': 10, 'N10': 6}, 'F5': {'N1': 13, 'N2': 12, 'N3': 10, 'N4': 8, 'N5': 7, 'N6': 6, 'N7': 2, 'N8': 3, 'N9': 4, 'N10': 5}, 'F6': {'N1': 12, 'N2': 11, 'N3': 8, 'N4': 7, 'N5': 6, 'N6': 5, 'N7': 5, 'N8': 4, 'N9': 3, 'N10': 2}}
    p = 2
    for i in facilities:
        if i not in distance or set(distance[i].keys()) != set(neighborhoods):
            raise ValueError(f'Distance data missing for facility {i} or neighborhoods mismatch.')
    if set(demand.keys()) != set(neighborhoods):
        raise ValueError('Demand data missing for some neighborhoods.')
    m = gp.Model('p_median')
    y = m.addVars(facilities, vtype=GRB.BINARY, name='')
    x = m.addVars(facilities, neighborhoods, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((demand[j] * distance[i][j] * x[i, j] for i in facilities for j in neighborhoods)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in facilities)) == 1 for j in neighborhoods), name='')
    m.addConstr(gp.quicksum((y[i] for i in facilities)) == p, name='open_facilities')
    m.addConstrs((x[i, j] <= y[i] for i in facilities for j in neighborhoods), name='')
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