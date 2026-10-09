from gurobipy import Model, GRB, quicksum
warehouses = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20']
f = {'W1': 2000, 'W2': 2500, 'W3': 1800, 'W4': 3200, 'W5': 1500, 'W6': 4000, 'W7': 2800, 'W8': 1950, 'W9': 3500, 'W10': 2200}
K = {'W1': 1000, 'W2': 1500, 'W3': 1200, 'W4': 2000, 'W5': 800, 'W6': 2500, 'W7': 1800, 'W8': 1100, 'W9': 2100, 'W10': 1300}
d = {'C1': 800, 'C2': 600, 'C3': 500, 'C4': 700, 'C5': 450, 'C6': 950, 'C7': 350, 'C8': 850, 'C9': 400, 'C10': 750, 'C11': 900, 'C12': 550, 'C13': 650, 'C14': 820, 'C15': 480, 'C16': 920, 'C17': 320, 'C18': 780, 'C19': 520, 'C20': 680}
C_matrix = [[10, 15, 20, 11, 16, 18, 7, 12, 22, 9, 14, 19, 25, 13, 17, 6, 21, 15, 8, 10], [18, 12, 9, 14, 10, 5, 19, 23, 11, 16, 20, 8, 15, 22, 7, 13, 24, 17, 12, 6], [13, 17, 15, 8, 12, 21, 16, 10, 5, 24, 13, 22, 7, 19, 14, 18, 9, 25, 11, 16], [7, 22, 11, 16, 20, 8, 15, 19, 13, 25, 6, 14, 21, 9, 23, 17, 10, 18, 24, 5], [16, 9, 25, 13, 7, 10, 23, 14, 18, 21, 5, 17, 9, 24, 12, 20, 6, 15, 19, 11], [22, 6, 14, 19, 23, 11, 8, 17, 9, 12, 15, 24, 5, 20, 10, 25, 13, 7, 18, 16], [8, 25, 17, 9, 14, 22, 11, 6, 16, 20, 18, 13, 24, 5, 19, 12, 23, 10, 7, 15], [19, 11, 7, 21, 15, 24, 13, 16, 20, 8, 17, 10, 12, 23, 5, 14, 22, 9, 16, 25], [12, 20, 5, 23, 17, 14, 9, 25, 18, 11, 16, 21, 10, 7, 24, 15, 19, 6, 13, 22], [25, 14, 22, 5, 19, 12, 24, 7, 15, 17, 23, 6, 16, 10, 20, 9, 18, 11, 25, 14]]
c = {}
for (i, wi) in enumerate(warehouses):
    for (j, cj) in enumerate(customers):
        c[wi, cj] = C_matrix[i][j]
if len(f) != len(warehouses):
    raise ValueError('Mismatch in number of warehouse fixed costs')
if len(K) != len(warehouses):
    raise ValueError('Mismatch in number of warehouse capacities')
if len(d) != len(customers):
    raise ValueError('Mismatch in number of customer demands')
for wi in warehouses:
    for cj in customers:
        if (wi, cj) not in c:
            raise ValueError(f'Missing transportation cost for {wi}, {cj}')

def build_model():
    m = Model()
    m.setParam('MIPGap', 0.0001)
    y_vars = m.addVars(warehouses, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(warehouses, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
    for cj in customers:
        m.addConstr(quicksum((x_vars[wi, cj] for wi in warehouses)) == d[cj], name='')
    for wi in warehouses:
        m.addConstr(quicksum((x_vars[wi, cj] for cj in customers)) <= K[wi] * y_vars[wi], name='')
    m.setObjective(quicksum((f[wi] * y_vars[wi] for wi in warehouses)) + quicksum((c[wi, cj] * x_vars[wi, cj] for wi in warehouses for cj in customers)), GRB.MINIMIZE)
    m.update()
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for wi in warehouses:
            print(f'y[{wi}] {y_vars[wi].X}')
        for wi in warehouses:
            for cj in customers:
                print(f'x[{wi},{cj}] {x_vars[wi, cj].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_model()