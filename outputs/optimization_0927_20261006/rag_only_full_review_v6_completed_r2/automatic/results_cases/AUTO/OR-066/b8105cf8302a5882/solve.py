from gurobipy import Model, GRB, quicksum
F = ['S1', 'S2']
C = ['C1', 'C2']
fixed_cost = {'S1': 105.97, 'S2': 85.31}
transportation_cost = {('S1', 'C1'): 2358.39, ('S1', 'C2'): 1492.08, ('S2', 'C1'): 0.07, ('S2', 'C2'): 52.32}
demand = {'C1': 144, 'C2': 216}
if set(fixed_cost.keys()) != set(F):
    raise ValueError('fixed_cost keys do not match F')
if set(demand.keys()) != set(C):
    raise ValueError('demand keys do not match C')
if set(transportation_cost.keys()) != set(((i, j) for i in F for j in C)):
    raise ValueError('transportation_cost keys do not match F x C')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(F, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(F, C, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_cost[i] * y_vars[i] for i in F)) + quicksum((transportation_cost[i, j] * x_vars[i, j] for i in F for j in C)), GRB.MINIMIZE)
    for j in C:
        m.addConstr(quicksum((x_vars[i, j] for i in F)) == demand[j], name='demand_' + j)
    total_demand = sum((demand[j] for j in C))
    for i in F:
        m.addConstr(quicksum((x_vars[i, j] for j in C)) <= total_demand * y_vars[i], name='supply_' + i)
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status', m.Status)