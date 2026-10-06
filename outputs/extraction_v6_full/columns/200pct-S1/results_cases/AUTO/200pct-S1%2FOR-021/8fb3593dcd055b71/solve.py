import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    suppliers = ['S1', 'S2', 'S3', 'S4']
    customers = ['C1', 'C2', 'C3', 'C4']
    demand = {'C1': 94, 'C2': 39, 'C3': 65, 'C4': 435}
    supply_capacity = {'S1': 2531, 'S2': 20, 'S3': 210, 'S4': 241}
    cost = {'S1': {'C1': 543.756480860856, 'C2': 23.685276141764653, 'C3': 23.676386730773032, 'C4': 447.75143678673766}, 'S2': {'C1': 883.9151090405642, 'C2': 0.04977684765576961, 'C3': 0.0350986687216299, 'C4': 44.45588531711622}, 'S3': {'C1': 537.3456896658107, 'C2': 23.769274659075112, 'C3': 498.95659249465467, 'C4': 440.60737890439776}, 'S4': {'C1': 1791.493192397229, 'C2': 68.21633865655126, 'C3': 1432.4837339656747, 'C4': 1527.7635425462734}}
    for i in suppliers:
        if i not in cost or i not in supply_capacity:
            raise ValueError(f'Missing cost or supply_capacity for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('BrewCo_Transportation')
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='x')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='demand')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='supply')
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