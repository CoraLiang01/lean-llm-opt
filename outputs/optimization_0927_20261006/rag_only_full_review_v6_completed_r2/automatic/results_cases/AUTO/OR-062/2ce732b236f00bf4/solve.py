from gurobipy import Model, GRB, quicksum
suppliers = {1: 'MOUNT AYR', 2: 'WAUKEE', 3: 'WAVERLY', 4: 'PELLA', 5: 'DES MOINES'}
stores = {1: 'CLARINDA', 2: 'FORT MADISON', 3: 'SIOUX CITY', 4: 'TOLEDO', 5: 'BANCROFT'}
fixed_cost = {1: 96.58, 2: 94.06, 3: 94.37, 4: 82.88, 5: 94.96}
transport_cost = {1: {1: 694.68, 2: 17.48, 3: 20.07, 4: 199.02, 5: 1685.53}, 2: {1: 15.13, 2: 1.5, 3: 1.43, 4: 27.88, 5: 90.69}, 3: {1: 2.34, 2: 349.34, 3: 246.6, 4: 41.3, 5: 78.73}, 4: {1: 1181.6, 2: 1458.53, 3: 1646.36, 4: 1924.55, 5: 38.93}, 5: {1: 1030.8, 2: 43.48, 3: 932.43, 4: 55.39, 5: 103.84}}
demand = {1: 2397, 2: 1889, 3: 2518, 4: 3218, 5: 1813}
if set(fixed_cost.keys()) != set(suppliers.keys()):
    raise ValueError('Fixed cost keys do not match supplier keys.')
if set(transport_cost.keys()) != set(suppliers.keys()):
    raise ValueError('Transport cost supplier keys do not match supplier keys.')
for i in suppliers:
    if set(transport_cost[i].keys()) != set(stores.keys()):
        raise ValueError(f'Transport cost store keys for supplier {i} do not match store keys.')
if set(demand.keys()) != set(stores.keys()):
    raise ValueError('Demand keys do not match store keys.')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(suppliers.keys(), vtype=GRB.BINARY, name='')
    x_vars = m.addVars(suppliers.keys(), stores.keys(), vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)) + quicksum((transport_cost[i][j] * x_vars[i, j] for i in suppliers for j in stores)), GRB.MINIMIZE)
    m.addConstrs((quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
    m.addConstrs((x_vars[i, j] <= demand[j] * y_vars[i] for i in suppliers for j in stores), name='')
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status', m.Status)