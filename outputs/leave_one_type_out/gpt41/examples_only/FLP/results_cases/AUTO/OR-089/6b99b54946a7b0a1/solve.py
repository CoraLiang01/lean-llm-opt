import gurobipy as gp
from gurobipy import GRB

def solve_service_center_location():
    I = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8', 'SC9', 'SC10']
    J = ['C%d' % i for i in range(1, 16)]
    fi = {'SC1': 385.1, 'SC2': 546.3, 'SC3': 485.2, 'SC4': 448.1, 'SC5': 324.1, 'SC6': 323.9, 'SC7': 296.5, 'SC8': 522.7, 'SC9': 448.7, 'SC10': 478.7}
    cij_table = [[15.1, 21.2, 14.9, 18.8, 22.9, 16.8, 16.5, 9.4, 16.1, 17.3], [13.4, 16.3, 20.2, 19.6, 20.9, 22.1, 16.9, 9.4, 13.8, 11.7], [15.2, 18.8, 14.7, 21.7, 18.1, 18.6, 12.3, 11.2, 11.9, 20.4], [16.8, 19.1, 18.3, 18.8, 23.1, 15.7, 13.1, 8.6, 15.6, 22.2], [13.4, 18.6, 20.8, 19.8, 22.1, 18.1, 16.7, 12.1, 11.4, 18.2], [12.5, 22.5, 15.5, 14.9, 21.6, 21.3, 16.1, 10.7, 11.9, 14.6], [12.1, 17.1, 19.8, 18.6, 22.1, 20.7, 20.5, 12.2, 15.4, 18.7], [12.3, 15.7, 17.9, 21.3, 22.7, 15.3, 16.6, 11.4, 14.1, 20.1], [16.3, 21.3, 17.6, 20.8, 21.8, 17.2, 15.5, 12.6, 19.9, 19.1], [12.1, 18.7, 14.4, 20.1, 22.7, 14.1, 18.1, 11.4, 18.1, 17.4], [16.7, 18.7, 15.7, 19.9, 24.2, 18.7, 14.2, 13.1, 14.7, 16.1], [11.3, 23.8, 15.5, 17.3, 23.2, 17.7, 16.8, 14.5, 15.8, 17.8], [15.1, 20.5, 15.1, 18.4, 20.6, 17.9, 14.5, 8.5, 14.9, 13.9], [8.3, 20.7, 14.7, 20.4, 20.6, 14.8, 14.2, 11.5, 14.1, 15.1], [12.1, 16.3, 16.4, 15.1, 21.3, 19.1, 19.5, 16.7, 11.1, 18.7]]
    cij = {}
    for j_idx, j in enumerate(J):
        for i_idx, i in enumerate(I):
            cij[i, j] = cij_table[j_idx][i_idx]
    if set(fi.keys()) != set(I):
        raise ValueError('fi keys do not match I')
    for i in I:
        for j in J:
            if (i, j) not in cij:
                raise ValueError(f'Missing cij for {(i, j)}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(I, J, vtype=GRB.BINARY, lb=0, ub=1, name='')
    m.setObjective(gp.quicksum((fi[i] * y[i] for i in I)) + gp.quicksum((cij[i, j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == 1, name='assign_%s' % j)
    for i in I:
        for j in J:
            m.addConstr(x[i, j] <= y[i], name='link_%s_%s' % (i, j))
    for i in I:
        m.addConstr(gp.quicksum((x[i, j] for j in J)) <= 4 * y[i], name='cap_%s' % i)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_service_center_location()