import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8', 'SC9', 'SC10']
    J = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
    f = {'SC1': 385.1, 'SC2': 546.3, 'SC3': 485.2, 'SC4': 448.1, 'SC5': 324.1, 'SC6': 323.9, 'SC7': 296.5, 'SC8': 522.7, 'SC9': 448.7, 'SC10': 478.7}
    c = {'SC1': {'C1': 15.1, 'C2': 13.4, 'C3': 15.2, 'C4': 16.8, 'C5': 13.4, 'C6': 12.5, 'C7': 12.1, 'C8': 12.3, 'C9': 16.3, 'C10': 12.1, 'C11': 16.7, 'C12': 11.3, 'C13': 15.1, 'C14': 8.3, 'C15': 12.1}, 'SC2': {'C1': 21.2, 'C2': 16.3, 'C3': 18.8, 'C4': 19.1, 'C5': 18.6, 'C6': 22.5, 'C7': 17.1, 'C8': 15.7, 'C9': 21.3, 'C10': 18.7, 'C11': 18.7, 'C12': 23.8, 'C13': 20.5, 'C14': 20.7, 'C15': 16.3}, 'SC3': {'C1': 14.9, 'C2': 20.2, 'C3': 14.7, 'C4': 18.3, 'C5': 20.8, 'C6': 15.5, 'C7': 19.8, 'C8': 17.9, 'C9': 17.6, 'C10': 14.4, 'C11': 15.7, 'C12': 15.5, 'C13': 15.1, 'C14': 14.7, 'C15': 16.4}, 'SC4': {'C1': 18.8, 'C2': 19.6, 'C3': 21.7, 'C4': 18.8, 'C5': 19.8, 'C6': 14.9, 'C7': 18.6, 'C8': 21.3, 'C9': 20.8, 'C10': 20.1, 'C11': 19.9, 'C12': 17.3, 'C13': 18.4, 'C14': 20.4, 'C15': 15.1}, 'SC5': {'C1': 22.9, 'C2': 20.9, 'C3': 18.1, 'C4': 23.1, 'C5': 22.1, 'C6': 21.6, 'C7': 22.1, 'C8': 22.7, 'C9': 21.8, 'C10': 22.7, 'C11': 24.2, 'C12': 23.2, 'C13': 20.6, 'C14': 20.6, 'C15': 21.3}, 'SC6': {'C1': 16.8, 'C2': 22.1, 'C3': 18.6, 'C4': 15.7, 'C5': 18.1, 'C6': 21.3, 'C7': 20.7, 'C8': 15.3, 'C9': 17.2, 'C10': 14.1, 'C11': 18.7, 'C12': 17.7, 'C13': 17.9, 'C14': 14.8, 'C15': 19.1}, 'SC7': {'C1': 16.5, 'C2': 16.9, 'C3': 12.3, 'C4': 13.1, 'C5': 16.7, 'C6': 16.1, 'C7': 20.5, 'C8': 16.6, 'C9': 15.5, 'C10': 18.1, 'C11': 14.2, 'C12': 16.8, 'C13': 14.5, 'C14': 14.2, 'C15': 19.5}, 'SC8': {'C1': 9.4, 'C2': 9.4, 'C3': 11.2, 'C4': 8.6, 'C5': 12.1, 'C6': 10.7, 'C7': 12.2, 'C8': 11.4, 'C9': 12.6, 'C10': 11.4, 'C11': 13.1, 'C12': 14.5, 'C13': 8.5, 'C14': 11.5, 'C15': 16.7}, 'SC9': {'C1': 16.1, 'C2': 13.8, 'C3': 11.9, 'C4': 15.6, 'C5': 11.4, 'C6': 11.9, 'C7': 15.4, 'C8': 14.1, 'C9': 19.9, 'C10': 18.1, 'C11': 14.7, 'C12': 15.8, 'C13': 14.9, 'C14': 14.1, 'C15': 11.1}, 'SC10': {'C1': 17.3, 'C2': 11.7, 'C3': 20.4, 'C4': 22.2, 'C5': 18.2, 'C6': 14.6, 'C7': 18.7, 'C8': 20.1, 'C9': 19.1, 'C10': 17.4, 'C11': 16.1, 'C12': 17.8, 'C13': 13.9, 'C14': 15.1, 'C15': 18.7}}
    for i in I:
        if i not in f:
            raise ValueError(f'Missing fixed cost for {i}')
        if i not in c:
            raise ValueError(f'Missing service cost row for {i}')
        for j in J:
            if j not in c[i]:
                raise ValueError(f'Missing service cost for ({i},{j})')
    m = gp.Model('ServiceCentreAssignment')
    m.setParam('MIPGap', 0.0001)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars(I, J, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == 1 for j in J), name='')
    m.addConstrs((x[i, j] <= y[i] for i in I for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= 4 * y[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()