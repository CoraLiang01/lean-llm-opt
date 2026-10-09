import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3', 'S4', 'S5']
stores = ['D1', 'D2', 'D3', 'D4', 'D5']
demand = {'D1': 428, 'D2': 217, 'D3': 214, 'D4': 380, 'D5': 254}
supply_capacity = {'S1': 428, 'S2': 217, 'S3': 214, 'S4': 380, 'S5': 254}
cost = {'S1': {'D1': 269.39105880208, 'D2': 1.4537335390934, 'D3': 99.603453457566, 'D4': 26.640781663098, 'D5': 9.5376889568809}, 'S2': {'D1': 9.2918468767852, 'D2': 10.87477843707, 'D3': 144.52609291615, 'D4': 11.420133077898, 'D5': 153.17568199278}, 'S3': {'D1': 9.674584301671, 'D2': 2.6191650959688, 'D3': 100.82422491687, 'D4': 3.2121910887917, 'D5': 133.84933961242}, 'S4': {'D1': 270.5749848001, 'D2': 32.50253586, 'D3': 4.684209809647, 'D4': 1.5682269686547, 'D5': 9.58927599}, 'S5': {'D1': 226.03319106758, 'D2': 8.6691619808265, 'D3': 65.476813169684, 'D4': 9.06876525846, 'D5': 202.65015316426}}
for i in warehouses:
    if i not in cost:
        raise ValueError(f'Missing cost row for warehouse {i}')
    for j in stores:
        if j not in cost[i]:
            raise ValueError(f'Missing cost entry for warehouse {i}, store {j}')
for j in stores:
    if j not in demand:
        raise ValueError(f'Missing demand for store {j}')
for i in warehouses:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for warehouse {i}')

def build_and_solve():
    model = gp.Model('GreenMart_Transportation')
    x_vars = model.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
    model.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
    model.Params.MIPGap = 0.0001
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = build_and_solve()