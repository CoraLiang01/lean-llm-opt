from gurobipy import Model, GRB, quicksum
S = ['Supplier1', 'Supplier2', 'Supplier3', 'Supplier4', 'Supplier5']
C = ['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6']
demand_c = {'Customer1': 70, 'Customer2': 80, 'Customer3': 60, 'Customer4': 90, 'Customer5': 85, 'Customer6': 95}
supply_capacity_s = {'Supplier1': 200, 'Supplier2': 250, 'Supplier3': 230, 'Supplier4': 220, 'Supplier5': 210}
transportation_cost = {('Supplier1', 'Customer1'): 2, ('Supplier1', 'Customer2'): 3, ('Supplier1', 'Customer3'): 1, ('Supplier1', 'Customer4'): 2, ('Supplier1', 'Customer5'): 3, ('Supplier1', 'Customer6'): 2, ('Supplier2', 'Customer1'): 1, ('Supplier2', 'Customer2'): 2, ('Supplier2', 'Customer3'): 3, ('Supplier2', 'Customer4'): 2, ('Supplier2', 'Customer5'): 3, ('Supplier2', 'Customer6'): 2, ('Supplier3', 'Customer1'): 3, ('Supplier3', 'Customer2'): 1, ('Supplier3', 'Customer3'): 2, ('Supplier3', 'Customer4'): 3, ('Supplier3', 'Customer5'): 2, ('Supplier3', 'Customer6'): 3, ('Supplier4', 'Customer1'): 2, ('Supplier4', 'Customer2'): 3, ('Supplier4', 'Customer3'): 2, ('Supplier4', 'Customer4'): 1, ('Supplier4', 'Customer5'): 3, ('Supplier4', 'Customer6'): 4, ('Supplier5', 'Customer1'): 3, ('Supplier5', 'Customer2'): 2, ('Supplier5', 'Customer3'): 3, ('Supplier5', 'Customer4'): 3, ('Supplier5', 'Customer5'): 2, ('Supplier5', 'Customer6'): 3}
for s in S:
    for c in C:
        if (s, c) not in transportation_cost:
            raise ValueError(f'Missing transportation cost for ({s}, {c})')
for c in C:
    if c not in demand_c:
        raise ValueError(f'Missing demand for {c}')
for s in S:
    if s not in supply_capacity_s:
        raise ValueError(f'Missing supply capacity for {s}')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(S, C, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(quicksum((transportation_cost[s, c] * x_vars[s, c] for s in S for c in C)), GRB.MINIMIZE)
    for c in C:
        m.addConstr(quicksum((x_vars[s, c] for s in S)) == demand_c[c], name='')
    for s in S:
        m.addConstr(quicksum((x_vars[s, c] for c in C)) <= supply_capacity_s[s], name='')
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')