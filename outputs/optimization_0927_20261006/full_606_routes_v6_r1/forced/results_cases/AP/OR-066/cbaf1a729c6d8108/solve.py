import gurobipy as gp
from gurobipy import GRB
suppliers = ['S1', 'S2']
customers = ['C1', 'C2']
fixed_costs = {'S1': 105.97, 'S2': 85.31}
demands = {'C1': 144, 'C2': 216}
transportation_costs = {'S1': {'C1': 2358.39, 'C2': 1492.08}, 'S2': {'C1': 0.07, 'C2': 52.32}}
for i in suppliers:
    if i not in fixed_costs:
        raise ValueError(f'Missing fixed cost for supplier {i}')
    if i not in transportation_costs:
        raise ValueError(f'Missing transportation costs for supplier {i}')
    for j in customers:
        if j not in transportation_costs[i]:
            raise ValueError(f'Missing transportation cost for supplier {i} to customer {j}')
for j in customers:
    if j not in demands:
        raise ValueError(f'Missing demand for customer {j}')
m = gp.Model('Supplier_Activation_Distribution')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(suppliers, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y_vars[i] for i in suppliers)) + gp.quicksum((transportation_costs[i][j] * x_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demands[j] for j in customers), name='')
m.addConstrs((x_vars[i, j] <= demands[j] * y_vars[i] for i in suppliers for j in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')