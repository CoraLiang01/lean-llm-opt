from gurobipy import Model, GRB

def colorado_motor_vehicle_sales():
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    dealerships = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    fixed_cost = {'S1': 100.64, 'S2': 98.72, 'S3': 100.18, 'S4': 96.58, 'S5': 95.75, 'S6': 99.06, 'S7': 101.78, 'S8': 93.86}
    demand = {'C1': 4742532000, 'C2': 1600594000, 'C3': 5086889000, 'C4': 1027326000, 'C5': 11926044000, 'C6': 9058407000, 'C7': 5344367000, 'C8': 677201000, 'C9': 3236493000}
    transportation_cost = {('S1', 'C1'): 1091.04, ('S1', 'C2'): 85.72, ('S1', 'C3'): 99.08, ('S1', 'C4'): 747.35, ('S1', 'C5'): 893.86, ('S1', 'C6'): 23.65, ('S1', 'C7'): 15.11, ('S1', 'C8'): 15.03, ('S1', 'C9'): 497.88, ('S2', 'C1'): 58.88, ('S2', 'C2'): 1617.16, ('S2', 'C3'): 1786.44, ('S2', 'C4'): 951.81, ('S2', 'C5'): 56.45, ('S2', 'C6'): 642.77, ('S2', 'C7'): 16.69, ('S2', 'C8'): 0.63, ('S2', 'C9'): 11.2, ('S3', 'C1'): 110.47, ('S3', 'C2'): 0.04, ('S3', 'C3'): 38.89, ('S3', 'C4'): 1397.95, ('S3', 'C5'): 2361.45, ('S3', 'C6'): 107.62, ('S3', 'C7'): 1598.5, ('S3', 'C8'): 76.41, ('S3', 'C9'): 1382.84, ('S4', 'C1'): 1458.85, ('S4', 'C2'): 1049.27, ('S4', 'C3'): 597.32, ('S4', 'C4'): 1731.9, ('S4', 'C5'): 69.09, ('S4', 'C6'): 1227.17, ('S4', 'C7'): 1187.55, ('S4', 'C8'): 1017.16, ('S4', 'C9'): 52.15, ('S5', 'C1'): 0.38, ('S5', 'C2'): 2315.52, ('S5', 'C3'): 1313.06, ('S5', 'C4'): 1253.71, ('S5', 'C5'): 50.24, ('S5', 'C6'): 29.19, ('S5', 'C7'): 60.17, ('S5', 'C8'): 1077.35, ('S5', 'C9'): 70.11, ('S6', 'C1'): 58.2, ('S6', 'C2'): 1395.81, ('S6', 'C3'): 84.6, ('S6', 'C4'): 830.64, ('S6', 'C5'): 1003.86, ('S6', 'C6'): 631.17, ('S6', 'C7'): 31.13, ('S6', 'C8'): 1.4, ('S6', 'C9'): 246.24, ('S7', 'C1'): 1255.23, ('S7', 'C2'): 1382.31, ('S7', 'C3'): 78.79, ('S7', 'C4'): 829.02, ('S7', 'C5'): 67.31, ('S7', 'C6'): 877.35, ('S7', 'C7'): 185.28, ('S7', 'C8'): 221.98, ('S7', 'C9'): 0.05, ('S8', 'C1'): 1990.09, ('S8', 'C2'): 1.23, ('S8', 'C3'): 38.97, ('S8', 'C4'): 1396.35, ('S8', 'C5'): 112.54, ('S8', 'C6'): 107.54, ('S8', 'C7'): 1596.74, ('S8', 'C8'): 76.32, ('S8', 'C9'): 1183.79}
    for s in suppliers:
        if s not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {s}')
        for c in dealerships:
            if (s, c) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for supplier {s}, dealership {c}')
    for c in dealerships:
        if c not in demand:
            raise ValueError(f'Missing demand for dealership {c}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    y = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(suppliers, dealerships, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(sum((fixed_cost[s] * y[s] for s in suppliers)) + sum((transportation_cost[s, c] * x[s, c] for s in suppliers for c in dealerships)), GRB.MINIMIZE)
    for c in dealerships:
        m.addConstr(sum((x[s, c] for s in suppliers)) == demand[c], name='d')
    for s in suppliers:
        for c in dealerships:
            m.addConstr(x[s, c] <= demand[c] * y[s], name='link')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = colorado_motor_vehicle_sales()