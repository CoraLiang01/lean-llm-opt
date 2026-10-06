from gurobipy import Model, GRB

def solve_facility_location():
    warehouses = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    customers = ['C%d' % i for i in range(1, 21)]
    f = {'W1': 2000, 'W2': 2500, 'W3': 1800, 'W4': 3200, 'W5': 1500, 'W6': 4000, 'W7': 2800, 'W8': 1950, 'W9': 3500, 'W10': 2200}
    s = {'W1': 1000, 'W2': 1500, 'W3': 1200, 'W4': 2000, 'W5': 800, 'W6': 2500, 'W7': 1800, 'W8': 1100, 'W9': 2100, 'W10': 1300}
    d = {'C1': 800, 'C2': 600, 'C3': 500, 'C4': 700, 'C5': 450, 'C6': 950, 'C7': 350, 'C8': 850, 'C9': 400, 'C10': 750, 'C11': 900, 'C12': 550, 'C13': 650, 'C14': 820, 'C15': 480, 'C16': 920, 'C17': 320, 'C18': 780, 'C19': 520, 'C20': 680}
    c_matrix = [[10, 15, 20, 11, 16, 18, 7, 12, 22, 9, 14, 19, 25, 13, 17, 6, 21, 15, 8, 10], [18, 12, 9, 14, 10, 5, 19, 23, 11, 16, 20, 8, 15, 22, 7, 13, 24, 17, 12, 6], [13, 17, 15, 8, 12, 21, 16, 10, 5, 24, 13, 22, 7, 19, 14, 18, 9, 25, 11, 16], [7, 22, 11, 16, 20, 8, 15, 19, 13, 25, 6, 14, 21, 9, 23, 17, 10, 18, 24, 5], [16, 9, 25, 13, 7, 10, 23, 14, 18, 21, 5, 17, 9, 24, 12, 20, 6, 15, 19, 11], [22, 6, 14, 19, 23, 11, 8, 17, 9, 12, 15, 24, 5, 20, 10, 25, 13, 7, 18, 16], [8, 25, 17, 9, 14, 22, 11, 6, 16, 20, 18, 13, 24, 5, 19, 12, 23, 10, 7, 15], [19, 11, 7, 21, 15, 24, 13, 16, 20, 8, 17, 10, 12, 23, 5, 14, 22, 9, 16, 25], [12, 20, 5, 23, 17, 14, 9, 25, 18, 11, 16, 21, 10, 7, 24, 15, 19, 6, 13, 22], [25, 14, 22, 5, 19, 12, 24, 7, 15, 17, 23, 6, 16, 10, 20, 9, 18, 11, 25, 14]]
    c = {}
    for wi, w in enumerate(warehouses):
        for cj, cst in enumerate(c_matrix[wi]):
            c[w, customers[cj]] = cst
    if set(f.keys()) != set(warehouses):
        raise ValueError('Fixed cost data missing for some warehouses')
    if set(s.keys()) != set(warehouses):
        raise ValueError('Capacity data missing for some warehouses')
    if set(d.keys()) != set(customers):
        raise ValueError('Demand data missing for some customers')
    for w in warehouses:
        for cst in customers:
            if (w, cst) not in c:
                raise ValueError(f'Transportation cost missing for ({w},{cst})')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    x = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(sum((f[w] * y[w] for w in warehouses)) + sum((c[w, cst] * x[w, cst] for w in warehouses for cst in customers)), GRB.MINIMIZE)
    for cst in customers:
        m.addConstr(sum((x[w, cst] for w in warehouses)) == d[cst], name='')
    for w in warehouses:
        m.addConstr(sum((x[w, cst] for cst in customers)) <= s[w] * y[w], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print('Status', m.Status)
    return m
m = solve_facility_location()