import gurobipy as gp
from gurobipy import GRB
facilities = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
distribution_centers = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
facility_data = {'A1': {'FixedCost': 0, 'Capacity': 30}, 'A2': {'FixedCost': 175, 'Capacity': 10}, 'A3': {'FixedCost': 300, 'Capacity': 20}, 'A4': {'FixedCost': 375, 'Capacity': 30}, 'A5': {'FixedCost': 500, 'Capacity': 40}, 'A6': {'FixedCost': 200, 'Capacity': 20}, 'A7': {'FixedCost': 260, 'Capacity': 25}, 'A8': {'FixedCost': 220, 'Capacity': 30}, 'A9': {'FixedCost': 320, 'Capacity': 35}, 'A10': {'FixedCost': 280, 'Capacity': 20}, 'A11': {'FixedCost': 350, 'Capacity': 40}, 'A12': {'FixedCost': 420, 'Capacity': 25}, 'A13': {'FixedCost': 470, 'Capacity': 30}, 'A14': {'FixedCost': 520, 'Capacity': 50}, 'A15': {'FixedCost': 560, 'Capacity': 45}}
demands = {'B1': 30, 'B2': 25, 'B3': 20, 'B4': 35, 'B5': 25, 'B6': 30, 'B7': 25, 'B8': 30}
shipping_costs = {'A1': {'B1': 8, 'B2': 4, 'B3': 3, 'B4': 6, 'B5': 7, 'B6': 5, 'B7': 9, 'B8': 8}, 'A2': {'B1': 5, 'B2': 2, 'B3': 3, 'B4': 5, 'B5': 6, 'B6': 4, 'B7': 7, 'B8': 6}, 'A3': {'B1': 4, 'B2': 3, 'B3': 4, 'B4': 6, 'B5': 5, 'B6': 5, 'B7': 6, 'B8': 7}, 'A4': {'B1': 9, 'B2': 7, 'B3': 5, 'B4': 8, 'B5': 9, 'B6': 6, 'B7': 10, 'B8': 7}, 'A5': {'B1': 10, 'B2': 4, 'B3': 2, 'B4': 6, 'B5': 8, 'B6': 5, 'B7': 7, 'B8': 3}, 'A6': {'B1': 6, 'B2': 5, 'B3': 4, 'B4': 5, 'B5': 7, 'B6': 6, 'B7': 8, 'B8': 5}, 'A7': {'B1': 7, 'B2': 6, 'B3': 5, 'B4': 4, 'B5': 6, 'B6': 7, 'B7': 9, 'B8': 6}, 'A8': {'B1': 5, 'B2': 4, 'B3': 6, 'B4': 3, 'B5': 5, 'B6': 6, 'B7': 7, 'B8': 6}, 'A9': {'B1': 8, 'B2': 7, 'B3': 6, 'B4': 7, 'B5': 9, 'B6': 8, 'B7': 10, 'B8': 7}, 'A10': {'B1': 6, 'B2': 5, 'B3': 7, 'B4': 4, 'B5': 6, 'B6': 5, 'B7': 7, 'B8': 5}, 'A11': {'B1': 9, 'B2': 6, 'B3': 4, 'B4': 6, 'B5': 8, 'B6': 7, 'B7': 9, 'B8': 6}, 'A12': {'B1': 7, 'B2': 5, 'B3': 6, 'B4': 5, 'B5': 6, 'B6': 5, 'B7': 8, 'B8': 5}, 'A13': {'B1': 8, 'B2': 6, 'B3': 5, 'B4': 6, 'B5': 7, 'B6': 6, 'B7': 8, 'B8': 7}, 'A14': {'B1': 9, 'B2': 5, 'B3': 3, 'B4': 5, 'B5': 7, 'B6': 4, 'B7': 6, 'B8': 4}, 'A15': {'B1': 10, 'B2': 6, 'B3': 4, 'B4': 5, 'B5': 8, 'B6': 5, 'B7': 7, 'B8': 5}}
for i in facilities:
    if i not in facility_data:
        raise ValueError(f'Missing facility data for {i}')
    if i not in shipping_costs:
        raise ValueError(f'Missing shipping costs for {i}')
    for j in distribution_centers:
        if j not in shipping_costs[i]:
            raise ValueError(f'Missing shipping cost for {i},{j}')
for j in distribution_centers:
    if j not in demands:
        raise ValueError(f'Missing demand for {j}')
m = gp.Model('ElectroTech_Facility_Location')
y_vars = m.addVars(facilities, vtype=GRB.BINARY, name='')
x_vars = m.addVars(facilities, distribution_centers, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((facility_data[i]['FixedCost'] * y_vars[i] for i in facilities)) + gp.quicksum((shipping_costs[i][j] * x_vars[i, j] for i in facilities for j in distribution_centers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in facilities)) == demands[j] for j in distribution_centers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in distribution_centers)) <= facility_data[i]['Capacity'] * y_vars[i] for i in facilities), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')