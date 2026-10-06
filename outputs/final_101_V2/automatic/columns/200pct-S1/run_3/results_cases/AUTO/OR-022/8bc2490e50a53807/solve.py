import gurobipy as gp
from gurobipy import GRB
suppliers = ['S1', 'S2', 'S3', 'S4', 'S5']
customers = ['C1', 'C2', 'C3', 'C4', 'C5']
demand = {'C1': 143, 'C2': 6, 'C3': 10, 'C4': 25, 'C5': 3}
fixed_cost = {'S1': 97.65, 'S2': 99.76, 'S3': 100.76, 'S4': 105.32, 'S5': 98.88}
cost = {'S1': {'C1': 150.74, 'C2': 0.02, 'C3': 49.13, 'C4': 2080.15, 'C5': 426.4}, 'S2': {'C1': 233.05, 'C2': 97.73, 'C3': 49.84, 'C4': 1982.39, 'C5': 23.96}, 'S3': {'C1': 55.68, 'C2': 935.61, 'C3': 4.03, 'C4': 73.09, 'C5': 525.32}, 'S4': {'C1': 1483.82, 'C2': 1801.08, 'C3': 112.16, 'C4': 816.05, 'C5': 107.01}, 'S5': {'C1': 1119.47, 'C2': 884.31, 'C3': 0.08, 'C4': 1544.95, 'C5': 543.67}}
M = sum(demand.values())
for i in suppliers:
    if i not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for supplier {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for supplier {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
m = gp.Model('Superstore_FLP')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')