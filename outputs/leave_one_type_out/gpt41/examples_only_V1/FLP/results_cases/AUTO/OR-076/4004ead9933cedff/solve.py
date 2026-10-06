import gurobipy as gp
from gurobipy import GRB

def solve_warehouse_location():
    I = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10']
    J = ['C%d' % i for i in range(1, 21)]
    f = {'W1': 2000, 'W2': 2500, 'W3': 1800, 'W4': 3200, 'W5': 1500, 'W6': 4000, 'W7': 2800, 'W8': 1950, 'W9': 3500, 'W10': 2200}
    cap = {'W1': 1000, 'W2': 1500, 'W3': 1200, 'W4': 2000, 'W5': 800, 'W6': 2500, 'W7': 1800, 'W8': 1100, 'W9': 2100, 'W10': 1300}
    d = {'C1': 800, 'C2': 600, 'C3': 500, 'C4': 700, 'C5': 450, 'C6': 950, 'C7': 350, 'C8': 850, 'C9': 400, 'C10': 750, 'C11': 900, 'C12': 550, 'C13': 650, 'C14': 820, 'C15': 480, 'C16': 920, 'C17': 320, 'C18': 780, 'C19': 520, 'C20': 680}
    c = {'W1': {'C1': 10, 'C2': 15, 'C3': 20, 'C4': 11, 'C5': 16, 'C6': 18, 'C7': 7, 'C8': 12, 'C9': 22, 'C10': 9, 'C11': 14, 'C12': 19, 'C13': 25, 'C14': 13, 'C15': 17, 'C16': 6, 'C17': 21, 'C18': 15, 'C19': 8, 'C20': 10}, 'W2': {'C1': 18, 'C2': 12, 'C3': 9, 'C4': 14, 'C5': 10, 'C6': 5, 'C7': 19, 'C8': 23, 'C9': 11, 'C10': 16, 'C11': 20, 'C12': 8, 'C13': 15, 'C14': 22, 'C15': 7, 'C16': 13, 'C17': 24, 'C18': 17, 'C19': 12, 'C20': 6}, 'W3': {'C1': 13, 'C2': 17, 'C3': 15, 'C4': 8, 'C5': 12, 'C6': 21, 'C7': 16, 'C8': 10, 'C9': 5, 'C10': 24, 'C11': 13, 'C12': 22, 'C13': 7, 'C14': 19, 'C15': 14, 'C16': 18, 'C17': 9, 'C18': 25, 'C19': 11, 'C20': 16}, 'W4': {'C1': 7, 'C2': 22, 'C3': 11, 'C4': 16, 'C5': 20, 'C6': 8, 'C7': 15, 'C8': 19, 'C9': 13, 'C10': 25, 'C11': 6, 'C12': 14, 'C13': 21, 'C14': 9, 'C15': 23, 'C16': 17, 'C17': 10, 'C18': 18, 'C19': 24, 'C20': 5}, 'W5': {'C1': 16, 'C2': 9, 'C3': 25, 'C4': 13, 'C5': 7, 'C6': 10, 'C7': 23, 'C8': 14, 'C9': 18, 'C10': 21, 'C11': 5, 'C12': 17, 'C13': 9, 'C14': 24, 'C15': 12, 'C16': 20, 'C17': 6, 'C18': 15, 'C19': 19, 'C20': 11}, 'W6': {'C1': 22, 'C2': 6, 'C3': 14, 'C4': 19, 'C5': 23, 'C6': 11, 'C7': 8, 'C8': 17, 'C9': 9, 'C10': 12, 'C11': 15, 'C12': 24, 'C13': 5, 'C14': 20, 'C15': 10, 'C16': 25, 'C17': 13, 'C18': 7, 'C19': 18, 'C20': 16}, 'W7': {'C1': 8, 'C2': 25, 'C3': 17, 'C4': 9, 'C5': 14, 'C6': 22, 'C7': 11, 'C8': 6, 'C9': 16, 'C10': 20, 'C11': 18, 'C12': 13, 'C13': 24, 'C14': 5, 'C15': 19, 'C16': 12, 'C17': 23, 'C18': 10, 'C19': 7, 'C20': 15}, 'W8': {'C1': 19, 'C2': 11, 'C3': 7, 'C4': 21, 'C5': 15, 'C6': 24, 'C7': 13, 'C8': 16, 'C9': 20, 'C10': 8, 'C11': 17, 'C12': 10, 'C13': 12, 'C14': 23, 'C15': 5, 'C16': 14, 'C17': 22, 'C18': 9, 'C19': 16, 'C20': 25}, 'W9': {'C1': 12, 'C2': 20, 'C3': 5, 'C4': 23, 'C5': 17, 'C6': 14, 'C7': 9, 'C8': 25, 'C9': 18, 'C10': 11, 'C11': 16, 'C12': 21, 'C13': 10, 'C14': 7, 'C15': 24, 'C16': 15, 'C17': 19, 'C18': 6, 'C19': 13, 'C20': 22}, 'W10': {'C1': 25, 'C2': 14, 'C3': 22, 'C4': 5, 'C5': 19, 'C6': 12, 'C7': 24, 'C8': 7, 'C9': 15, 'C10': 17, 'C11': 23, 'C12': 6, 'C13': 16, 'C14': 10, 'C15': 20, 'C16': 9, 'C17': 18, 'C18': 11, 'C19': 25, 'C20': 14}}
    if set(f.keys()) != set(I):
        raise ValueError('Mismatch in warehouse opening cost keys and I')
    if set(cap.keys()) != set(I):
        raise ValueError('Mismatch in warehouse capacity keys and I')
    if set(d.keys()) != set(J):
        raise ValueError('Mismatch in customer demand keys and J')
    if set(c.keys()) != set(I):
        raise ValueError('Mismatch in c keys and I')
    for i in I:
        if set(c[i].keys()) != set(J):
            raise ValueError(f'Mismatch in c[{i}] keys and J')
    m = gp.Model()
    y = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == d[j], name='demand_%s' % j)
    for i in I:
        m.addConstr(gp.quicksum((x[i, j] for j in J)) <= cap[i] * y[i], name='cap_%s' % i)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for i in I:
            print(y[i].VarName, y[i].X)
        for i in I:
            for j in J:
                print(x[i, j].VarName, x[i, j].X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_warehouse_location()