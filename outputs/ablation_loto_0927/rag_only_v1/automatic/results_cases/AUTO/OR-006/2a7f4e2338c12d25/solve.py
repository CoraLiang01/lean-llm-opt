import gurobipy as gp
from gurobipy import GRB

def solve_transportation():
    S = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    C = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    demand = {'C1': 45, 'C2': 23, 'C3': 94, 'C4': 92, 'C5': 57, 'C6': 52, 'C7': 23, 'C8': 99, 'C9': 99, 'C10': 77}
    supply = {'S1': 127, 'S2': 236, 'S3': 168, 'S4': 115, 'S5': 280, 'S6': 179, 'S7': 135, 'S8': 263, 'S9': 283, 'S10': 476}
    cost = {'S1': {'C1': 2077.0586725, 'C2': 0.0, 'C3': 54.3352648, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629, 'C7': 0.0, 'C8': 0.0, 'C9': 169.33026927, 'C10': 0.0}, 'S2': {'C1': 2077.0586725, 'C2': 0.0, 'C3': 1141.0405609, 'C4': 0.0, 'C5': 0.0, 'C6': 651.11123325, 'C7': 0.0, 'C8': 0.0, 'C9': 8.063346156, 'C10': 0.0}, 'S3': {'C1': 79.9210296, 'C2': 474.24509131, 'C3': 1477.0676289, 'C4': 22.58309959, 'C5': 474.24509131, 'C6': 41.10659696, 'C7': 474.24509131, 'C8': 474.24509131, 'C9': 624.1625395, 'C10': 474.24509131}, 'S4': {'C1': 1659.3369291, 'C2': 57.20541469, 'C3': 186.15190481, 'C4': 1201.3137084, 'C5': 1029.6974644, 'C6': 41.82210594, 'C7': 57.20541469, 'C8': 1201.3137084, 'C9': 884.56338707, 'C10': 1029.6974644}, 'S5': {'C1': 1297.2567041, 'C2': 77.76629131, 'C3': 24.26760228, 'C4': 1399.7932436, 'C5': 77.76629131, 'C6': 53.91161728, 'C7': 1399.7932436, 'C8': 77.76629131, 'C9': 1255.115148, 'C10': 1399.7932436}, 'S6': {'C1': 1998.9090659, 'C2': 985.31654357, 'C3': 2.854168689, 'C4': 1149.5359675, 'C5': 985.31654357, 'C6': 730.69236477, 'C7': 54.73980798, 'C8': 985.31654357, 'C9': 46.80310221, 'C10': 1149.5359675}, 'S7': {'C1': 1780.336005, 'C2': 0.0, 'C3': 1141.0405609, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629, 'C7': 0.0, 'C8': 0.0, 'C9': 8.063346156, 'C10': 0.0}, 'S8': {'C1': 75.40935896, 'C2': 1338.1987291, 'C3': 21.39134599, 'C4': 74.34437384, 'C5': 74.34437384, 'C6': 937.35062391, 'C7': 1338.1987291, 'C8': 1338.1987291, 'C9': 1392.1186581, 'C10': 1338.1987291}, 'S9': {'C1': 98.90755583, 'C2': 0.0, 'C3': 978.03476648, 'C4': 0.0, 'C5': 0.0, 'C6': 651.11123325, 'C7': 0.0, 'C8': 0.0, 'C9': 169.33026927, 'C10': 0.0}, 'S10': {'C1': 2077.0586725, 'C2': 0.0, 'C3': 54.3352648, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629, 'C7': 0.0, 'C8': 0.0, 'C9': 145.1402308, 'C10': 0.0}}
    if set(demand.keys()) != set(C):
        raise ValueError('Demand data missing or extra for some customers.')
    if set(supply.keys()) != set(S):
        raise ValueError('Supply data missing or extra for some warehouses.')
    for i in S:
        if i not in cost:
            raise ValueError(f'Cost data missing for warehouse {i}.')
        if set(cost[i].keys()) != set(C):
            raise ValueError(f'Cost data for warehouse {i} missing or extra for some customers.')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(S, C, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in S for j in C)), GRB.MINIMIZE)
    for j in C:
        m.addConstr(gp.quicksum((x[i, j] for i in S)) == demand[j], name='')
    for i in S:
        m.addConstr(gp.quicksum((x[i, j] for j in C)) <= supply[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in S:
            for j in C:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_transportation()