from gurobipy import Model, GRB, quicksum
warehouses = ['S1', 'S2', 'S3', 'S4', 'S5']
stores = ['D1', 'D2', 'D3', 'D4', 'D5']
demand = {'D1': 428, 'D2': 217, 'D3': 214, 'D4': 380, 'D5': 254}
supply_capacity = {'S1': 428, 'S2': 217, 'S3': 214, 'S4': 380, 'S5': 254}
cost = {('S1', 'D1'): 269.3910588020795, ('S1', 'D2'): 1.4537335390933939, ('S1', 'D3'): 99.60345345756605, ('S1', 'D4'): 26.64078166309837, ('S1', 'D5'): 9.537688956880922, ('S2', 'D1'): 9.291846876785183, ('S2', 'D2'): 10.874778437070223, ('S2', 'D3'): 144.52609291614627, ('S2', 'D4'): 11.420133077898234, ('S2', 'D5'): 153.1756819927813, ('S3', 'D1'): 9.674584301671008, ('S3', 'D2'): 2.6191650959687944, ('S3', 'D3'): 100.8242249168735, ('S3', 'D4'): 3.2121910887916876, ('S3', 'D5'): 133.8493396124168, ('S4', 'D1'): 270.57498480010247, ('S4', 'D2'): 32.50253586, ('S4', 'D3'): 4.6842098096469815, ('S4', 'D4'): 1.5682269686546804, ('S4', 'D5'): 9.58927599, ('S5', 'D1'): 226.0331910675782, ('S5', 'D2'): 8.669161980826471, ('S5', 'D3'): 65.47681316968448, ('S5', 'D4'): 9.068765258459958, ('S5', 'D5'): 202.65015316425533}
for i in warehouses:
    for j in stores:
        if (i, j) not in cost:
            raise ValueError(f'Missing cost coefficient for ({i},{j})')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand for {j}')
for i in warehouses:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for {i}')

def build_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(quicksum((cost[i, j] * x_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    for j in stores:
        m.addConstr(quicksum((x_vars[i, j] for i in warehouses)) == demand[j], name='')
    for i in warehouses:
        m.addConstr(quicksum((x_vars[i, j] for j in stores)) <= supply_capacity[i], name='')
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for i in warehouses:
        for j in stores:
            v = m.getVarByName(f'x[{i},{j}]') if m.getVarByName(f'x[{i},{j}]') else m.getVarByName(f'{i},{j}')
            if v is None:
                v = m.getVarByName(f'{i}_{j}')
            if v is None:
                v = m.getVars()[warehouses.index(i) * len(stores) + stores.index(j)]
            print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')