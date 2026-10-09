import gurobipy as gp
from gurobipy import GRB

def solve_warehouse_location():
    warehouses = ['W%d' % i for i in range(1, 11)]
    customers = ['C%d' % j for j in range(1, 21)]
    f_list = [2000, 2500, 1800, 3200, 1500, 4000, 2800, 1950, 3500, 2200]
    s_list = [1000, 1500, 1200, 2000, 800, 2500, 1800, 1100, 2100, 1300]
    d_list = [800, 600, 500, 700, 450, 950, 350, 850, 400, 750, 900, 550, 650, 820, 480, 920, 320, 780, 520, 680]
    c_matrix = [[10, 15, 20, 11, 16, 18, 7, 12, 22, 9, 14, 19, 25, 13, 17, 6, 21, 15, 8, 10], [18, 12, 9, 14, 10, 5, 19, 23, 11, 16, 20, 8, 15, 22, 7, 13, 24, 17, 12, 6], [13, 17, 15, 8, 12, 21, 16, 10, 5, 24, 13, 22, 7, 19, 14, 18, 9, 25, 11, 16], [7, 22, 11, 16, 20, 8, 15, 19, 13, 25, 6, 14, 21, 9, 23, 17, 10, 18, 24, 5], [16, 9, 25, 13, 7, 10, 23, 14, 18, 21, 5, 17, 9, 24, 12, 20, 6, 15, 19, 11], [22, 6, 14, 19, 23, 11, 8, 17, 9, 12, 15, 24, 5, 20, 10, 25, 13, 7, 18, 16], [8, 25, 17, 9, 14, 22, 11, 6, 16, 20, 18, 13, 24, 5, 19, 12, 23, 10, 7, 15], [19, 11, 7, 21, 15, 24, 13, 16, 20, 8, 17, 10, 12, 23, 5, 14, 22, 9, 16, 25], [12, 20, 5, 23, 17, 14, 9, 25, 18, 11, 16, 21, 10, 7, 24, 15, 19, 6, 13, 22], [25, 14, 22, 5, 19, 12, 24, 7, 15, 17, 23, 6, 16, 10, 20, 9, 18, 11, 25, 14]]
    f = {warehouses[i]: f_list[i] for i in range(10)}
    s = {warehouses[i]: s_list[i] for i in range(10)}
    d = {customers[j]: d_list[j] for j in range(20)}
    c = {}
    for (i, wi) in enumerate(warehouses):
        for (j, cj) in enumerate(customers):
            c[wi, cj] = c_matrix[i][j]
    if len(f) != len(warehouses):
        raise ValueError('Mismatch in number of warehouse opening costs')
    if len(s) != len(warehouses):
        raise ValueError('Mismatch in number of warehouse capacities')
    if len(d) != len(customers):
        raise ValueError('Mismatch in number of customer demands')
    for wi in warehouses:
        for cj in customers:
            if (wi, cj) not in c:
                raise ValueError(f'Missing transportation cost for ({wi}, {cj})')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((f[wi] * y[wi] for wi in warehouses)) + gp.quicksum((c[wi, cj] * x[wi, cj] for wi in warehouses for cj in customers)), GRB.MINIMIZE)
    for cj in customers:
        m.addConstr(gp.quicksum((x[wi, cj] for wi in warehouses)) == d[cj], name='')
    for wi in warehouses:
        m.addConstr(gp.quicksum((x[wi, cj] for cj in customers)) <= s[wi] * y[wi], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for wi in warehouses:
            print(y[wi].VarName, y[wi].X)
        for wi in warehouses:
            for cj in customers:
                print(x[wi, cj].VarName, x[wi, cj].X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_warehouse_location()