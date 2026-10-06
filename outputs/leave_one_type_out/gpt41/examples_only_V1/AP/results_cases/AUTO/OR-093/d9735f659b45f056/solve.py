import gurobipy as gp
from gurobipy import GRB

def solve_assignment():
    machines = list(range(1, 13))
    tasks = list(range(1, 13))
    C_matrix = [[167.4, 98.6, 189.4, 119.6, 182.0, 145.1, 185.4, 94.8, 122.3, 123.3, 96.1, 90.3], [156.2, 88.7, 187.3, 124.7, 173.2, 144.3, 179.0, 91.5, 115.1, 119.5, 100.1, 88.6], [184.3, 121.0, 216.6, 140.0, 196.2, 168.8, 205.6, 114.2, 133.3, 144.5, 116.0, 107.7], [157.9, 92.9, 185.1, 120.3, 175.1, 146.2, 180.8, 86.3, 111.6, 115.9, 98.1, 91.1], [175.6, 103.6, 204.5, 130.0, 192.8, 157.5, 194.2, 106.9, 129.9, 134.9, 105.8, 98.6], [166.8, 107.0, 199.2, 130.4, 183.6, 159.5, 187.0, 98.2, 121.3, 126.2, 105.9, 101.8], [159.7, 93.2, 183.8, 113.0, 171.9, 139.1, 169.6, 85.1, 110.0, 116.7, 90.6, 85.2], [184.8, 115.9, 205.1, 138.6, 195.4, 160.1, 200.2, 108.5, 136.9, 140.0, 114.6, 103.9], [157.3, 86.2, 186.0, 113.9, 166.2, 136.8, 167.5, 78.8, 107.4, 114.5, 87.2, 78.6], [164.8, 97.8, 200.9, 125.8, 188.9, 151.2, 187.7, 99.5, 119.5, 132.1, 101.1, 98.4], [164.0, 92.2, 186.2, 115.7, 174.5, 143.0, 175.9, 92.3, 114.0, 121.2, 93.7, 91.2], [151.7, 76.7, 179.5, 109.5, 160.6, 128.4, 170.2, 74.4, 103.7, 110.4, 83.7, 75.2]]
    C = {}
    for i in machines:
        C[i] = {}
        for j in tasks:
            C[i][j] = C_matrix[i - 1][j - 1]
    if len(C) != 12 or any((len(C[i]) != 12 for i in machines)):
        raise ValueError('Cost matrix dimensions do not match the sets of machines and tasks.')
    for i in machines:
        for j in tasks:
            if i not in C or j not in C[i]:
                raise ValueError(f'Missing cost coefficient for machine {i}, task {j}.')
    m = gp.Model()
    x = m.addVars(machines, tasks, vtype=GRB.BINARY, lb=0, ub=1, name='')
    for i in machines:
        m.addConstr(gp.quicksum((x[i, j] for j in tasks)) == 1, name='mc')
    for j in tasks:
        m.addConstr(gp.quicksum((x[i, j] for i in machines)) == 1, name='tc')
    m.setObjective(gp.quicksum((C[i][j] * x[i, j] for i in machines for j in tasks)), GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in machines:
            for j in tasks:
                var = x[i, j]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_assignment()