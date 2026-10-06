import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    warehouses = ['S1', 'S2', 'S3', 'S4', 'S5']
    stores = ['D1', 'D2', 'D3', 'D4', 'D5']
    demand = {'D1': 428, 'D2': 217, 'D3': 214, 'D4': 380, 'D5': 254}
    supply_capacity = {'S1': 428, 'S2': 217, 'S3': 214, 'S4': 380, 'S5': 254}
    cost = {'S1': {'D1': 269.3910588, 'D2': 1.453733539, 'D3': 99.60345346, 'D4': 26.64078166, 'D5': 9.537688957}, 'S2': {'D1': 9.291846877, 'D2': 10.87477844, 'D3': 144.5260929, 'D4': 11.42013308, 'D5': 153.17568199}, 'S3': {'D1': 9.674584302, 'D2': 2.619165096, 'D3': 100.8242249, 'D4': 3.212191089, 'D5': 133.84933961}, 'S4': {'D1': 270.5749848, 'D2': 32.50253586, 'D3': 4.68420981, 'D4': 1.568226969, 'D5': 9.58927599}, 'S5': {'D1': 226.0331911, 'D2': 8.669161981, 'D3': 65.47681317, 'D4': 9.068765258, 'D5': 202.65015316}}
    for i in warehouses:
        if i not in cost or not isinstance(cost[i], dict):
            raise ValueError(f'Missing cost data for warehouse {i}')
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Missing cost data for warehouse {i}, store {j}')
    m = gp.Model('GreenMart_Transportation')
    x = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='x')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) >= demand[j] for j in stores), name='demand')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='supply')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()