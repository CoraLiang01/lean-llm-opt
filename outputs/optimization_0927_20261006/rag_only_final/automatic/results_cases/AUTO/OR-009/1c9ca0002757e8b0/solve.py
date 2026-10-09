from gurobipy import Model, GRB, quicksum
plants = ['S1', 'S2', 'S3', 'S4']
customers = ['C1', 'C2', 'C3', 'C4']
transportation_costs = {('S1', 'C1'): 543.756480860856, ('S1', 'C2'): 23.685276141764653, ('S1', 'C3'): 23.676386730773032, ('S1', 'C4'): 447.75143678673766, ('S2', 'C1'): 883.9151090405642, ('S2', 'C2'): 0.04977684765576961, ('S2', 'C3'): 0.0350986687216299, ('S2', 'C4'): 44.45588531711622, ('S3', 'C1'): 537.3456896658107, ('S3', 'C2'): 23.769274659075112, ('S3', 'C3'): 498.95659249465467, ('S3', 'C4'): 440.60737890439776, ('S4', 'C1'): 1791.493192397229, ('S4', 'C2'): 68.21633865655126, ('S4', 'C3'): 1432.4837339656747, ('S4', 'C4'): 1527.7635425462734}
customer_demand = {'C1': 94, 'C2': 39, 'C3': 65, 'C4': 435}
supply_capacity = {'S1': 2531, 'S2': 20, 'S3': 210, 'S4': 241}
for i in plants:
    for j in customers:
        if (i, j) not in transportation_costs:
            raise ValueError(f'Missing transportation cost for ({i},{j})')
for j in customers:
    if j not in customer_demand:
        raise ValueError(f'Missing demand for customer {j}')
for i in plants:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for plant {i}')

def build_model():
    model = Model()
    model.Params.MIPGap = 0.0001
    x_vars = model.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = quicksum((transportation_costs[i, j] * x_vars[i, j] for i in plants for j in customers))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        model.addConstr(quicksum((x_vars[i, j] for i in plants)) == customer_demand[j], name=f'dem_{j}')
    for i in plants:
        model.addConstr(quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name=f'sup_{i}')
    return model
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for i in plants:
        for j in customers:
            v = m.getVarByName(f'x[{i},{j}]')
            print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')