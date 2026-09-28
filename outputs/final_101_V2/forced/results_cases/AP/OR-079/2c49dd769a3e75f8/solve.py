import gurobipy as gp
from gurobipy import GRB
factories = [{'Facility': 'A1', 'FixedCost': 0, 'Capacity': 30}, {'Facility': 'A2', 'FixedCost': 175, 'Capacity': 10}, {'Facility': 'A3', 'FixedCost': 300, 'Capacity': 20}, {'Facility': 'A4', 'FixedCost': 375, 'Capacity': 30}, {'Facility': 'A5', 'FixedCost': 500, 'Capacity': 40}, {'Facility': 'A6', 'FixedCost': 200, 'Capacity': 20}, {'Facility': 'A7', 'FixedCost': 260, 'Capacity': 25}, {'Facility': 'A8', 'FixedCost': 220, 'Capacity': 30}, {'Facility': 'A9', 'FixedCost': 320, 'Capacity': 35}, {'Facility': 'A10', 'FixedCost': 280, 'Capacity': 20}, {'Facility': 'A11', 'FixedCost': 350, 'Capacity': 40}, {'Facility': 'A12', 'FixedCost': 420, 'Capacity': 25}, {'Facility': 'A13', 'FixedCost': 470, 'Capacity': 30}, {'Facility': 'A14', 'FixedCost': 520, 'Capacity': 50}, {'Facility': 'A15', 'FixedCost': 560, 'Capacity': 45}]
distribution_centers = [{'Destination': 'B1', 'Demand': 30}, {'Destination': 'B2', 'Demand': 25}, {'Destination': 'B3', 'Demand': 20}, {'Destination': 'B4', 'Demand': 35}, {'Destination': 'B5', 'Demand': 25}, {'Destination': 'B6', 'Demand': 30}, {'Destination': 'B7', 'Demand': 25}, {'Destination': 'B8', 'Demand': 30}]
shipping_costs = {'A1': {'B1': 8, 'B2': 4, 'B3': 3, 'B4': 6, 'B5': 7, 'B6': 5, 'B7': 9, 'B8': 8}, 'A2': {'B1': 5, 'B2': 2, 'B3': 3, 'B4': 5, 'B5': 6, 'B6': 4, 'B7': 7, 'B8': 6}, 'A3': {'B1': 4, 'B2': 3, 'B3': 4, 'B4': 6, 'B5': 5, 'B6': 5, 'B7': 6, 'B8': 7}, 'A4': {'B1': 9, 'B2': 7, 'B3': 5, 'B4': 8, 'B5': 9, 'B6': 6, 'B7': 10, 'B8': 7}, 'A5': {'B1': 10, 'B2': 4, 'B3': 2, 'B4': 6, 'B5': 8, 'B6': 5, 'B7': 7, 'B8': 3}, 'A6': {'B1': 6, 'B2': 5, 'B3': 4, 'B4': 5, 'B5': 7, 'B6': 6, 'B7': 8, 'B8': 5}, 'A7': {'B1': 7, 'B2': 6, 'B3': 5, 'B4': 4, 'B5': 6, 'B6': 7, 'B7': 9, 'B8': 6}, 'A8': {'B1': 5, 'B2': 4, 'B3': 6, 'B4': 3, 'B5': 5, 'B6': 6, 'B7': 7, 'B8': 6}, 'A9': {'B1': 8, 'B2': 7, 'B3': 6, 'B4': 7, 'B5': 9, 'B6': 8, 'B7': 10, 'B8': 7}, 'A10': {'B1': 6, 'B2': 5, 'B3': 7, 'B4': 4, 'B5': 6, 'B6': 5, 'B7': 7, 'B8': 5}, 'A11': {'B1': 9, 'B2': 6, 'B3': 4, 'B4': 6, 'B5': 8, 'B6': 7, 'B7': 9, 'B8': 6}, 'A12': {'B1': 7, 'B2': 5, 'B3': 6, 'B4': 5, 'B5': 6, 'B6': 5, 'B7': 8, 'B8': 5}, 'A13': {'B1': 8, 'B2': 6, 'B3': 5, 'B4': 6, 'B5': 7, 'B6': 6, 'B7': 8, 'B8': 7}, 'A14': {'B1': 9, 'B2': 5, 'B3': 3, 'B4': 5, 'B5': 7, 'B6': 4, 'B7': 6, 'B8': 4}, 'A15': {'B1': 10, 'B2': 6, 'B3': 4, 'B4': 5, 'B5': 8, 'B6': 5, 'B7': 7, 'B8': 5}}
factory_ids = [f['Facility'] for f in factories]
dc_ids = [d['Destination'] for d in distribution_centers]
F = {f['Facility']: f['FixedCost'] for f in factories}
K = {f['Facility']: f['Capacity'] for f in factories}
D = {d['Destination']: d['Demand'] for d in distribution_centers}
c = shipping_costs
for i in factory_ids:
    if i not in c:
        raise ValueError(f'Missing shipping costs for factory {i}')
    for j in dc_ids:
        if j not in c[i]:
            raise ValueError(f'Missing shipping cost for factory {i} to DC {j}')
    if i not in F or i not in K:
        raise ValueError(f'Missing fixed cost or capacity for factory {i}')
for j in dc_ids:
    if j not in D:
        raise ValueError(f'Missing demand for DC {j}')
m = gp.Model('ElectroTech_Facility_Location')
y = m.addVars(factory_ids, vtype=GRB.BINARY, name='')
x = m.addVars(factory_ids, dc_ids, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((F[i] * y[i] for i in factory_ids)) + gp.quicksum((c[i][j] * x[i, j] for i in factory_ids for j in dc_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in factory_ids)) == D[j] for j in dc_ids), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in dc_ids)) <= K[i] * y[i] for i in factory_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')