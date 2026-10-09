import gurobipy as gp
from gurobipy import GRB
areas = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12']
locations = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']
area_demand = {'A1': 25, 'A2': 35, 'A3': 40, 'A4': 30, 'A5': 50, 'A6': 45, 'A7': 20, 'A8': 55, 'A9': 60, 'A10': 30, 'A11': 42, 'A12': 38}
distance = {'L1': {'A1': 2, 'A2': 3, 'A3': 4, 'A4': 8, 'A5': 9, 'A6': 10, 'A7': 13, 'A8': 14, 'A9': 15, 'A10': 12, 'A11': 11, 'A12': 10}, 'L2': {'A1': 3, 'A2': 2, 'A3': 3, 'A4': 7, 'A5': 8, 'A6': 9, 'A7': 12, 'A8': 13, 'A9': 14, 'A10': 11, 'A11': 10, 'A12': 9}, 'L3': {'A1': 8, 'A2': 7, 'A3': 5, 'A4': 2, 'A5': 3, 'A6': 4, 'A7': 8, 'A8': 9, 'A9': 11, 'A10': 7, 'A11': 6, 'A12': 7}, 'L4': {'A1': 9, 'A2': 8, 'A3': 6, 'A4': 3, 'A5': 2, 'A6': 3, 'A7': 7, 'A8': 8, 'A9': 10, 'A10': 6, 'A11': 5, 'A12': 6}, 'L5': {'A1': 13, 'A2': 12, 'A3': 10, 'A4': 8, 'A5': 7, 'A6': 6, 'A7': 2, 'A8': 3, 'A9': 4, 'A10': 5, 'A11': 6, 'A12': 7}, 'L6': {'A1': 14, 'A2': 13, 'A3': 11, 'A4': 9, 'A5': 8, 'A6': 7, 'A7': 3, 'A8': 2, 'A9': 3, 'A10': 4, 'A11': 5, 'A12': 6}, 'L7': {'A1': 11, 'A2': 10, 'A3': 8, 'A4': 7, 'A5': 6, 'A6': 5, 'A7': 6, 'A8': 5, 'A9': 4, 'A10': 2, 'A11': 3, 'A12': 2}}
number_of_facilities_to_open = 3
for i in locations:
    if i not in distance:
        raise ValueError(f'Missing distance data for location {i}')
    for j in areas:
        if j not in distance[i]:
            raise ValueError(f'Missing distance data for location {i}, area {j}')
for j in areas:
    if j not in area_demand:
        raise ValueError(f'Missing demand data for area {j}')
m = gp.Model('p_median')
x_vars = m.addVars(locations, areas, vtype=GRB.BINARY, name='')
y_vars = m.addVars(locations, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((area_demand[j] * distance[i][j] * x_vars[i, j] for i in locations for j in areas)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in locations)) == 1 for j in areas), name='')
m.addConstr(gp.quicksum((y_vars[i] for i in locations)) == number_of_facilities_to_open, name='open')
m.addConstrs((x_vars[i, j] <= y_vars[i] for i in locations for j in areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')