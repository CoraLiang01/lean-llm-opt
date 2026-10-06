import gurobipy as gp
from gurobipy import GRB

def colorado_motor_vehicle_facility_location():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    dealerships = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    fixed_cost = {'S1': 100.64, 'S2': 98.72, 'S3': 100.18, 'S4': 96.58, 'S5': 95.75, 'S6': 99.06, 'S7': 101.78, 'S8': 93.86}
    demand = {'C1': 4742532000, 'C2': 1600594000, 'C3': 5086889000, 'C4': 1027326000, 'C5': 11926044000, 'C6': 9058407000, 'C7': 5344367000, 'C8': 677201000, 'C9': 3236493000}
    transportation_cost = {'S1': {'C1': 1091.04, 'C2': 85.72, 'C3': 99.08, 'C4': 747.35, 'C5': 893.86, 'C6': 23.65, 'C7': 15.11, 'C8': 15.03, 'C9': 497.88}, 'S2': {'C1': 58.88, 'C2': 1617.16, 'C3': 1786.44, 'C4': 951.81, 'C5': 56.45, 'C6': 642.77, 'C7': 16.69, 'C8': 0.63, 'C9': 11.2}, 'S3': {'C1': 110.47, 'C2': 0.04, 'C3': 38.89, 'C4': 1397.95, 'C5': 2361.45, 'C6': 107.62, 'C7': 1598.5, 'C8': 76.41, 'C9': 1382.84}, 'S4': {'C1': 1458.85, 'C2': 1049.27, 'C3': 597.32, 'C4': 1731.9, 'C5': 69.09, 'C6': 1227.17, 'C7': 1187.55, 'C8': 1017.16, 'C9': 52.15}, 'S5': {'C1': 0.38, 'C2': 2315.52, 'C3': 1313.06, 'C4': 1253.71, 'C5': 50.24, 'C6': 29.19, 'C7': 60.17, 'C8': 1077.35, 'C9': 70.11}, 'S6': {'C1': 58.2, 'C2': 1395.81, 'C3': 84.6, 'C4': 830.64, 'C5': 1003.86, 'C6': 631.17, 'C7': 31.13, 'C8': 1.4, 'C9': 246.24}, 'S7': {'C1': 1255.23, 'C2': 1382.31, 'C3': 78.79, 'C4': 829.02, 'C5': 67.31, 'C6': 877.35, 'C7': 185.28, 'C8': 221.98, 'C9': 0.05}, 'S8': {'C1': 1990.09, 'C2': 1.23, 'C3': 38.97, 'C4': 1396.35, 'C5': 112.54, 'C6': 107.54, 'C7': 1596.74, 'C8': 76.32, 'C9': 1183.79}}
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in transportation_cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in dealerships:
            if j not in transportation_cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, dealership {j}')
    for j in dealerships:
        if j not in demand:
            raise ValueError(f'Missing demand for dealership {j}')
    m = gp.Model('colorado_motor_vehicle_facility_location')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    x = m.addVars(suppliers, dealerships, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_cost[i][j] * x[i, j] for i in suppliers for j in dealerships)), GRB.MINIMIZE)
    for j in dealerships:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name='')
    for i in suppliers:
        for j in dealerships:
            m.addConstr(x[i, j] <= demand[j] * y[i], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = colorado_motor_vehicle_facility_location()