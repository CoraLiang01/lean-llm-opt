import gurobipy as gp
from gurobipy import GRB
sources = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
destinations = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20']
supply = {'S1': 103, 'S2': 87, 'S3': 95, 'S4': 112, 'S5': 97, 'S6': 103, 'S7': 101, 'S8': 94, 'S9': 102, 'S10': 106}
demand = {'D1': 61, 'D2': 54, 'D3': 56, 'D4': 54, 'D5': 53, 'D6': 47, 'D7': 56, 'D8': 57, 'D9': 56, 'D10': 34, 'D11': 55, 'D12': 53, 'D13': 37, 'D14': 31, 'D15': 62, 'D16': 58, 'D17': 39, 'D18': 32, 'D19': 38, 'D20': 67}
cost = {'S1': {'D1': 2.75, 'D2': 2.6, 'D3': 2.9, 'D4': 1.7, 'D5': 1.85, 'D6': 1.98, 'D7': 2.74, 'D8': 6.2, 'D9': 5.75, 'D10': 6.44, 'D11': 5.2, 'D12': 4.54, 'D13': 5.39, 'D14': 4.34, 'D15': 8.28, 'D16': 8.87, 'D17': 9.03, 'D18': 8.49, 'D19': 9.66, 'D20': 10.91}, 'S2': {'D1': 5.24, 'D2': 4.87, 'D3': 4.72, 'D4': 4.12, 'D5': 4.26, 'D6': 4.55, 'D7': 4.45, 'D8': 3.88, 'D9': 3.24, 'D10': 4.24, 'D11': 3.13, 'D12': 3.16, 'D13': 3.67, 'D14': 2.43, 'D15': 5.83, 'D16': 6.51, 'D17': 6.68, 'D18': 6.1, 'D19': 7.2, 'D20': 8.4}, 'S3': {'D1': 5.54, 'D2': 5.16, 'D3': 5.09, 'D4': 4.37, 'D5': 4.57, 'D6': 4.78, 'D7': 4.89, 'D8': 3.84, 'D9': 3.08, 'D10': 4.42, 'D11': 3.37, 'D12': 3.53, 'D13': 3.98, 'D14': 2.7, 'D15': 5.84, 'D16': 6.48, 'D17': 6.68, 'D18': 6.14, 'D19': 7.24, 'D20': 8.31}, 'S4': {'D1': 4.51, 'D2': 4.17, 'D3': 4.01, 'D4': 3.36, 'D5': 3.59, 'D6': 3.73, 'D7': 3.68, 'D8': 4.39, 'D9': 3.93, 'D10': 4.71, 'D11': 3.55, 'D12': 3.15, 'D13': 3.84, 'D14': 2.6, 'D15': 6.51, 'D16': 7.03, 'D17': 7.27, 'D18': 6.69, 'D19': 7.89, 'D20': 9.04}, 'S5': {'D1': 4.84, 'D2': 4.74, 'D3': 4.77, 'D4': 3.8, 'D5': 3.96, 'D6': 4.1, 'D7': 4.52, 'D8': 4.82, 'D9': 4.03, 'D10': 5.35, 'D11': 4.25, 'D12': 4.14, 'D13': 4.73, 'D14': 3.46, 'D15': 6.89, 'D16': 7.5, 'D17': 7.67, 'D18': 7.2, 'D19': 8.2, 'D20': 9.35}, 'S6': {'D1': 9.13, 'D2': 8.32, 'D3': 7.78, 'D4': 7.9, 'D5': 8.13, 'D6': 8.45, 'D7': 7.47, 'D8': 3.04, 'D9': 4.25, 'D10': 2.19, 'D11': 3.34, 'D12': 4.07, 'D13': 3.3, 'D14': 4.24, 'D15': 2.77, 'D16': 2.63, 'D17': 2.97, 'D18': 2.56, 'D19': 3.32, 'D20': 5.09}, 'S7': {'D1': 8.64, 'D2': 7.86, 'D3': 7.34, 'D4': 7.32, 'D5': 7.51, 'D6': 7.83, 'D7': 7.03, 'D8': 1.9, 'D9': 3.14, 'D10': 1.43, 'D11': 2.64, 'D12': 3.64, 'D13': 2.92, 'D14': 3.51, 'D15': 2.41, 'D16': 2.74, 'D17': 2.98, 'D18': 2.46, 'D19': 3.51, 'D20': 5.08}, 'S8': {'D1': 9.25, 'D2': 8.41, 'D3': 7.81, 'D4': 8.13, 'D5': 8.32, 'D6': 8.6, 'D7': 7.49, 'D8': 3.71, 'D9': 5.0, 'D10': 2.88, 'D11': 3.78, 'D12': 4.27, 'D13': 3.49, 'D14': 4.63, 'D15': 3.62, 'D16': 3.34, 'D17': 3.59, 'D18': 3.29, 'D19': 3.79, 'D20': 5.54}, 'S9': {'D1': 10.3, 'D2': 9.58, 'D3': 8.89, 'D4': 9.11, 'D5': 9.32, 'D6': 9.69, 'D7': 8.57, 'D8': 4.11, 'D9': 5.29, 'D10': 3.49, 'D11': 4.59, 'D12': 5.28, 'D13': 4.47, 'D14': 5.51, 'D15': 3.22, 'D16': 2.59, 'D17': 2.72, 'D18': 2.68, 'D19': 2.82, 'D20': 4.47}, 'S10': {'D1': 7.85, 'D2': 7.08, 'D3': 6.57, 'D4': 6.72, 'D5': 6.88, 'D6': 7.18, 'D7': 6.21, 'D8': 2.34, 'D9': 3.56, 'D10': 1.52, 'D11': 2.18, 'D12': 2.78, 'D13': 2.01, 'D14': 3.0, 'D15': 3.47, 'D16': 3.62, 'D17': 3.89, 'D18': 3.41, 'D19': 4.38, 'D20': 5.98}}
for i in sources:
    if i not in supply:
        raise ValueError(f'Missing supply for {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for {i}')
    for j in destinations:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for {i},{j}')
for j in destinations:
    if j not in demand:
        raise ValueError(f'Missing demand for {j}')
m = gp.Model('Truck_Transport_Optimization')
x_vars = m.addVars(sources, destinations, lb=0, vtype=GRB.CONTINUOUS, name='')
t_vars = m.addVars(sources, destinations, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in sources for j in destinations)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in sources)) == demand[j] for j in destinations), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in destinations)) <= supply[i] for i in sources), name='')
m.addConstrs((x_vars[i, j] <= 10 * t_vars[i, j] for i in sources for j in destinations), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')