import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3', 'S4', 'S5']
stores = ['D1', 'D2', 'D3', 'D4', 'D5']
demand = {'D1': 428, 'D2': 217, 'D3': 214, 'D4': 380, 'D5': 254}
supply_capacity = {'S1': 428, 'S2': 217, 'S3': 214, 'S4': 380, 'S5': 254}
cost = {'S1': {'D1': 269.3910588020795, 'D2': 1.453733539093394, 'D3': 99.60345345756603, 'D4': 26.64078166309837, 'D5': 9.537688956880922}, 'S2': {'D1': 9.291846876785185, 'D2': 10.874778437070225, 'D3': 144.52609291614627, 'D4': 11.420133077898234, 'D5': 153.1756819927813}, 'S3': {'D1': 9.674584301671008, 'D2': 2.6191650959687944, 'D3': 100.8242249168735, 'D4': 3.212191088791688, 'D5': 133.8493396124168}, 'S4': {'D1': 270.57498480010247, 'D2': 32.50253586, 'D3': 4.6842098096469815, 'D4': 1.5682269686546804, 'D5': 9.58927599}, 'S5': {'D1': 226.0331910675782, 'D2': 8.669161980826471, 'D3': 65.47681316968448, 'D4': 9.068765258459958, 'D5': 202.65015316425533}}
for i in warehouses:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for warehouse {i}')
    if i not in cost:
        raise ValueError(f'Missing cost row for warehouse {i}')
    for j in stores:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for warehouse {i} to store {j}')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}')
m = gp.Model('GreenMart_Transportation')
x = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')