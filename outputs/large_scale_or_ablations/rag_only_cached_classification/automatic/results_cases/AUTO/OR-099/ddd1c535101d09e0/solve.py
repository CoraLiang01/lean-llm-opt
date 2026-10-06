import gurobipy as gp
from gurobipy import GRB

def solve_warehouse_location():
    warehouses = ['Wh 1', 'Wh 2', 'Wh 3', 'Wh 4', 'Wh 5', 'Wh 6', 'Wh 7', 'Wh 8', 'Wh 9', 'Wh 10', 'Wh 11']
    stores = ['Store 1', 'Store 2', 'Store 3', 'Store 4', 'Store 5', 'Store 6', 'Store 7', 'Store 8', 'Store 9', 'Store 10', 'Store 11']
    opening_costs = {'Wh 1': 3000, 'Wh 2': 3200, 'Wh 3': 3100, 'Wh 4': 2800, 'Wh 5': 3500, 'Wh 6': 2700, 'Wh 7': 2900, 'Wh 8': 3050, 'Wh 9': 3100, 'Wh 10': 2200, 'Wh 11': 2890}
    capacities = {'Wh 1': 180, 'Wh 2': 160, 'Wh 3': 200, 'Wh 4': 150, 'Wh 5': 170, 'Wh 6': 190, 'Wh 7': 160, 'Wh 8': 175, 'Wh 9': 170, 'Wh 10': 180, 'Wh 11': 190}
    demands = {'Store 1': 30, 'Store 2': 40, 'Store 3': 20, 'Store 4': 35, 'Store 5': 20, 'Store 6': 25, 'Store 7': 45, 'Store 8': 38, 'Store 9': 32, 'Store 10': 41, 'Store 11': 44}
    transportation_costs = {'Wh 1': {'Store 1': 12, 'Store 2': 11, 'Store 3': 14, 'Store 4': 15, 'Store 5': 17, 'Store 6': 13, 'Store 7': 12, 'Store 8': 16, 'Store 9': 16, 'Store 10': 14, 'Store 11': 15}, 'Wh 2': {'Store 1': 17, 'Store 2': 19, 'Store 3': 15, 'Store 4': 20, 'Store 5': 18, 'Store 6': 14, 'Store 7': 17, 'Store 8': 15, 'Store 9': 13, 'Store 10': 15, 'Store 11': 16}, 'Wh 3': {'Store 1': 13, 'Store 2': 14, 'Store 3': 12, 'Store 4': 14, 'Store 5': 16, 'Store 6': 15, 'Store 7': 11, 'Store 8': 14, 'Store 9': 16, 'Store 10': 18, 'Store 11': 17}, 'Wh 4': {'Store 1': 18, 'Store 2': 16, 'Store 3': 17, 'Store 4': 13, 'Store 5': 18, 'Store 6': 17, 'Store 7': 14, 'Store 8': 19, 'Store 9': 16, 'Store 10': 13, 'Store 11': 18}, 'Wh 5': {'Store 1': 10, 'Store 2': 13, 'Store 3': 12, 'Store 4': 19, 'Store 5': 15, 'Store 6': 11, 'Store 7': 12, 'Store 8': 14, 'Store 9': 12, 'Store 10': 15, 'Store 11': 17}, 'Wh 6': {'Store 1': 15, 'Store 2': 12, 'Store 3': 14, 'Store 4': 16, 'Store 5': 13, 'Store 6': 17, 'Store 7': 16, 'Store 8': 16, 'Store 9': 14, 'Store 10': 18, 'Store 11': 19}, 'Wh 7': {'Store 1': 14, 'Store 2': 13, 'Store 3': 15, 'Store 4': 17, 'Store 5': 12, 'Store 6': 13, 'Store 7': 14, 'Store 8': 15, 'Store 9': 12, 'Store 10': 16, 'Store 11': 14}, 'Wh 8': {'Store 1': 19, 'Store 2': 16, 'Store 3': 18, 'Store 4': 20, 'Store 5': 17, 'Store 6': 19, 'Store 7': 16, 'Store 8': 18, 'Store 9': 15, 'Store 10': 15, 'Store 11': 18}, 'Wh 9': {'Store 1': 17, 'Store 2': 18, 'Store 3': 12, 'Store 4': 14, 'Store 5': 16, 'Store 6': 15, 'Store 7': 14, 'Store 8': 17, 'Store 9': 21, 'Store 10': 15, 'Store 11': 18}, 'Wh 10': {'Store 1': 14, 'Store 2': 13, 'Store 3': 15, 'Store 4': 17, 'Store 5': 16, 'Store 6': 18, 'Store 7': 14, 'Store 8': 19, 'Store 9': 15, 'Store 10': 17, 'Store 11': 19}, 'Wh 11': {'Store 1': 15, 'Store 2': 13, 'Store 3': 16, 'Store 4': 17, 'Store 5': 11, 'Store 6': 13, 'Store 7': 14, 'Store 8': 15, 'Store 9': 19, 'Store 10': 21, 'Store 11': 13}}
    if set(opening_costs.keys()) != set(warehouses):
        raise ValueError('Mismatch in warehouse opening costs keys and warehouse list')
    if set(capacities.keys()) != set(warehouses):
        raise ValueError('Mismatch in warehouse capacities keys and warehouse list')
    if set(demands.keys()) != set(stores):
        raise ValueError('Mismatch in store demands keys and store list')
    if set(transportation_costs.keys()) != set(warehouses):
        raise ValueError('Mismatch in transportation_costs warehouse keys and warehouse list')
    for i in warehouses:
        if set(transportation_costs[i].keys()) != set(stores):
            raise ValueError(f'Mismatch in transportation_costs[{i}] store keys and store list')
    m = gp.Model()
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    x = m.addVars(warehouses, stores, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((opening_costs[i] * y[i] for i in warehouses)) + gp.quicksum((transportation_costs[i][j] * x[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demands[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= capacities[i] * y[i] for i in warehouses), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_warehouse_location()