import gurobipy as gp
from gurobipy import GRB

def solve_colorado_motor_vehicle_sales():
    suppliers = ['S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7', 'S_8']
    dealerships = ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9']
    fixed_cost = {'S_1': 100.64, 'S_2': 98.72, 'S_3': 100.18, 'S_4': 96.58, 'S_5': 95.75, 'S_6': 99.06, 'S_7': 101.78, 'S_8': 93.86}
    demand = {'C_1': 4742532000, 'C_2': 1600594000, 'C_3': 5086889000, 'C_4': 1027326000, 'C_5': 11926044000, 'C_6': 9058407000, 'C_7': 5344367000, 'C_8': 677201000, 'C_9': 3236493000}
    transportation_cost = {'S_1': {'C_1': 1091.04, 'C_2': 85.72, 'C_3': 99.08, 'C_4': 747.35, 'C_5': 893.86, 'C_6': 23.65, 'C_7': 15.11, 'C_8': 15.03, 'C_9': 497.88}, 'S_2': {'C_1': 58.88, 'C_2': 1617.16, 'C_3': 1786.44, 'C_4': 951.81, 'C_5': 56.45, 'C_6': 642.77, 'C_7': 16.69, 'C_8': 0.63, 'C_9': 11.2}, 'S_3': {'C_1': 110.47, 'C_2': 0.04, 'C_3': 38.89, 'C_4': 1397.95, 'C_5': 2361.45, 'C_6': 107.62, 'C_7': 1598.5, 'C_8': 76.41, 'C_9': 1382.84}, 'S_4': {'C_1': 1458.85, 'C_2': 1049.27, 'C_3': 597.32, 'C_4': 1731.9, 'C_5': 69.09, 'C_6': 1227.17, 'C_7': 1187.55, 'C_8': 1017.16, 'C_9': 52.15}, 'S_5': {'C_1': 0.38, 'C_2': 2315.52, 'C_3': 1313.06, 'C_4': 1253.71, 'C_5': 50.24, 'C_6': 29.19, 'C_7': 60.17, 'C_8': 1077.35, 'C_9': 70.11}, 'S_6': {'C_1': 58.2, 'C_2': 1395.81, 'C_3': 84.6, 'C_4': 830.64, 'C_5': 1003.86, 'C_6': 631.17, 'C_7': 31.13, 'C_8': 1.4, 'C_9': 246.24}, 'S_7': {'C_1': 1255.23, 'C_2': 1382.31, 'C_3': 78.79, 'C_4': 829.02, 'C_5': 67.31, 'C_6': 877.35, 'C_7': 185.28, 'C_8': 221.98, 'C_9': 0.05}, 'S_8': {'C_1': 1990.09, 'C_2': 1.23, 'C_3': 38.97, 'C_4': 1396.35, 'C_5': 112.54, 'C_6': 107.54, 'C_7': 1596.74, 'C_8': 76.32, 'C_9': 1183.79}}
    for s in suppliers:
        if s not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {s}')
        if s not in transportation_cost:
            raise ValueError(f'Missing transportation cost row for supplier {s}')
        for d in dealerships:
            if d not in transportation_cost[s]:
                raise ValueError(f'Missing transportation cost for supplier {s}, dealership {d}')
    for d in dealerships:
        if d not in demand:
            raise ValueError(f'Missing demand for dealership {d}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(suppliers, dealerships, vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gp.quicksum((fixed_cost[s] * y[s] for s in suppliers)) + gp.quicksum((transportation_cost[s][d] * x[s, d] for s in suppliers for d in dealerships))
    m.setObjective(obj, GRB.MINIMIZE)
    for d in dealerships:
        m.addConstr(gp.quicksum((x[s, d] for s in suppliers)) == demand[d], name='demand_' + d)
    for s in suppliers:
        for d in dealerships:
            m.addConstr(x[s, d] <= demand[d] * y[s], name='link_%s_%s' % (s, d))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_colorado_motor_vehicle_sales()