import gurobipy as gp
from gurobipy import GRB
plants = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
fixed_cost = {'F1': 11250, 'F2': 13480, 'F3': 14870, 'F4': 10290, 'F5': 16740, 'F6': 13960, 'F7': 12680, 'F8': 17890, 'F9': 10950, 'F10': 15320, 'F11': 11830, 'F12': 14110, 'F13': 15970, 'F14': 13140, 'F15': 10580}
capacity = {'F1': 101, 'F2': 124, 'F3': 139, 'F4': 86, 'F5': 157, 'F6': 133, 'F7': 118, 'F8': 162, 'F9': 92, 'F10': 144, 'F11': 107, 'F12': 129, 'F13': 151, 'F14': 113, 'F15': 85}
transportation_cost = {'F1': [7.8, 7.6, 6.7, 7.9, 8.1, 8.3, 7.3, 8.2, 8.1, 8.2, 7.3, 7.7, 6.7, 7.1, 7.9], 'F2': [5.3, 6, 5, 6.4, 5.9, 6.2, 5.6, 6.1, 6.3, 6.1, 5, 5.6, 5.3, 4.9, 6.3], 'F3': [7.2, 8.1, 7.4, 8.8, 8.5, 8.7, 7.7, 8.7, 8.9, 8.5, 7.2, 7.7, 7.1, 7.6, 8.4], 'F4': [7, 7.1, 6.5, 7.9, 7.4, 7.7, 6.7, 7.9, 7.8, 7.3, 6.8, 7, 6.5, 6.7, 7.6], 'F5': [3.5, 3.8, 2.9, 4.3, 3.6, 3.9, 3.2, 4.3, 4.5, 4, 3.2, 4, 2.9, 3.4, 3.9], 'F6': [8.2, 8.6, 7.9, 9.5, 8.5, 9.3, 8.5, 9.4, 9, 9.2, 8.1, 8.7, 7.9, 8.5, 9], 'F7': [6.9, 7.6, 6.8, 8.4, 8, 8, 7.6, 8, 8.1, 7.8, 6.9, 7.1, 7, 6.9, 7.5], 'F8': [6.9, 7.8, 7.1, 8.7, 8.6, 8.2, 7.2, 7.9, 8.4, 7.9, 7, 7.4, 6.8, 7.3, 8], 'F9': [3.5, 3.8, 2.8, 4.4, 4.2, 4.8, 3.8, 5, 4.5, 4.1, 3.2, 3.7, 3.7, 3.2, 4.5], 'F10': [5.2, 6.1, 5.1, 6.3, 6.1, 6, 5.6, 6.5, 6.2, 5.9, 5.3, 6.1, 5.1, 5.2, 6.2], 'F11': [5.2, 5.5, 4.5, 6.2, 5.7, 6.1, 5.1, 5.8, 5.7, 6.2, 5.2, 5.2, 4.5, 5.1, 5.4], 'F12': [7.8, 8.7, 7.6, 9, 8.6, 9, 8.5, 9.3, 9.3, 8.4, 7.9, 8.2, 7.4, 7.6, 8.7], 'F13': [6.7, 6.6, 6.1, 7.3, 7.1, 7.5, 6.7, 8, 7.6, 7.2, 6.3, 6.9, 6.2, 6, 7.2], 'F14': [7.5, 8.6, 7.6, 8.2, 8, 7.9, 7.5, 8.7, 8.8, 8.1, 7.2, 7.3, 7, 7, 8], 'F15': [5.1, 5.8, 4.6, 5.9, 6.5, 5.9, 5.2, 7, 7.1, 5.9, 5.1, 5.8, 5.4, 4.9, 6]}
demand = {'C1': 83, 'C2': 76, 'C3': 91, 'C4': 68, 'C5': 104, 'C6': 97, 'C7': 88, 'C8': 73, 'C9': 109, 'C10': 95, 'C11': 82, 'C12': 67, 'C13': 113, 'C14': 79, 'C15': 92}
for i in plants:
    if i not in fixed_cost or i not in capacity or i not in transportation_cost:
        raise ValueError(f'Missing plant data for {i}')
    if len(transportation_cost[i]) != len(customers):
        raise ValueError(f'Plant {i} transportation_cost length mismatch')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
cost = {}
for i in plants:
    cost[i] = {}
    for (idx, j) in enumerate(customers):
        cost[i][j] = transportation_cost[i][idx]
m = gp.Model('Plant_Location')
x_vars = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(plants, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in plants for j in customers)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in plants)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in plants)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= capacity[i] * y_vars[i] for i in plants), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')