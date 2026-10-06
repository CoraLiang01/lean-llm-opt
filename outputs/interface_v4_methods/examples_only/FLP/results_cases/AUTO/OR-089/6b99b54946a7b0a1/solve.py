import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    I = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8', 'SC9', 'SC10']
    J = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15']
    f = {'SC1': 385.1, 'SC2': 546.3, 'SC3': 485.2, 'SC4': 448.1, 'SC5': 324.1, 'SC6': 323.9, 'SC7': 296.5, 'SC8': 522.7, 'SC9': 448.7, 'SC10': 478.7}
    c = {'C1': {'SC1': 15.1, 'SC2': 21.2, 'SC3': 14.9, 'SC4': 18.8, 'SC5': 22.9, 'SC6': 16.8, 'SC7': 16.5, 'SC8': 9.4, 'SC9': 16.1, 'SC10': 17.3}, 'C2': {'SC1': 13.4, 'SC2': 16.3, 'SC3': 20.2, 'SC4': 19.6, 'SC5': 20.9, 'SC6': 22.1, 'SC7': 16.9, 'SC8': 9.4, 'SC9': 13.8, 'SC10': 11.7}, 'C3': {'SC1': 15.2, 'SC2': 18.8, 'SC3': 14.7, 'SC4': 21.7, 'SC5': 18.1, 'SC6': 18.6, 'SC7': 12.3, 'SC8': 11.2, 'SC9': 11.9, 'SC10': 20.4}, 'C4': {'SC1': 16.8, 'SC2': 19.1, 'SC3': 18.3, 'SC4': 18.8, 'SC5': 23.1, 'SC6': 15.7, 'SC7': 13.1, 'SC8': 8.6, 'SC9': 15.6, 'SC10': 22.2}, 'C5': {'SC1': 13.4, 'SC2': 18.6, 'SC3': 20.8, 'SC4': 19.8, 'SC5': 22.1, 'SC6': 18.1, 'SC7': 16.7, 'SC8': 12.1, 'SC9': 11.4, 'SC10': 18.2}, 'C6': {'SC1': 12.5, 'SC2': 22.5, 'SC3': 15.5, 'SC4': 14.9, 'SC5': 21.6, 'SC6': 21.3, 'SC7': 16.1, 'SC8': 10.7, 'SC9': 11.9, 'SC10': 14.6}, 'C7': {'SC1': 12.1, 'SC2': 17.1, 'SC3': 19.8, 'SC4': 18.6, 'SC5': 22.1, 'SC6': 20.7, 'SC7': 20.5, 'SC8': 12.2, 'SC9': 15.4, 'SC10': 18.7}, 'C8': {'SC1': 12.3, 'SC2': 15.7, 'SC3': 17.9, 'SC4': 21.3, 'SC5': 22.7, 'SC6': 15.3, 'SC7': 16.6, 'SC8': 11.4, 'SC9': 14.1, 'SC10': 20.1}, 'C9': {'SC1': 16.3, 'SC2': 21.3, 'SC3': 17.6, 'SC4': 20.8, 'SC5': 21.8, 'SC6': 17.2, 'SC7': 15.5, 'SC8': 12.6, 'SC9': 19.9, 'SC10': 19.1}, 'C10': {'SC1': 12.1, 'SC2': 18.7, 'SC3': 14.4, 'SC4': 20.1, 'SC5': 22.7, 'SC6': 14.1, 'SC7': 18.1, 'SC8': 11.4, 'SC9': 18.1, 'SC10': 17.4}, 'C11': {'SC1': 16.7, 'SC2': 18.7, 'SC3': 15.7, 'SC4': 19.9, 'SC5': 24.2, 'SC6': 18.7, 'SC7': 14.2, 'SC8': 13.1, 'SC9': 14.7, 'SC10': 16.1}, 'C12': {'SC1': 11.3, 'SC2': 23.8, 'SC3': 15.5, 'SC4': 17.3, 'SC5': 23.2, 'SC6': 17.7, 'SC7': 16.8, 'SC8': 14.5, 'SC9': 15.8, 'SC10': 17.8}, 'C13': {'SC1': 15.1, 'SC2': 20.5, 'SC3': 15.1, 'SC4': 18.4, 'SC5': 20.6, 'SC6': 17.9, 'SC7': 14.5, 'SC8': 8.5, 'SC9': 14.9, 'SC10': 13.9}, 'C14': {'SC1': 8.3, 'SC2': 20.7, 'SC3': 14.7, 'SC4': 20.4, 'SC5': 20.6, 'SC6': 14.8, 'SC7': 14.2, 'SC8': 11.5, 'SC9': 14.1, 'SC10': 15.1}, 'C15': {'SC1': 12.1, 'SC2': 16.3, 'SC3': 16.4, 'SC4': 15.1, 'SC5': 21.3, 'SC6': 19.1, 'SC7': 19.5, 'SC8': 16.7, 'SC9': 11.1, 'SC10': 18.7}}
    if set(f.keys()) != set(I):
        raise ValueError('Fixed cost keys do not match service centre set I')
    if set(c.keys()) != set(J):
        raise ValueError('Service cost customer keys do not match customer set J')
    for j in J:
        if set(c[j].keys()) != set(I):
            raise ValueError(f'Service cost for customer {j} does not cover all centres I')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(J, I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    m.setObjective(gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[j][i] * x[j, i] for j in J for i in I)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[j, i] for i in I)) == 1, name='assign_%s' % j)
    for j in J:
        for i in I:
            m.addConstr(x[j, i] <= y[i], name='link_%s_%s' % (j, i))
    for i in I:
        m.addConstr(gp.quicksum((x[j, i] for j in J)) <= 4 * y[i], name='cap_%s' % i)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print('Status', m.Status)
    return m
m = solve_problem()