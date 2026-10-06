import gurobipy as gp
from gurobipy import GRB

def solve_adidas_supplier_problem():
    I = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    J = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    fixed_cost = {'S1': 98.88, 'S2': 99.73, 'S3': 94.01, 'S4': 93.77, 'S5': 107.59, 'S6': 112.65}
    demand = {'C1': 216, 'C2': 216, 'C3': 216, 'C4': 144, 'C5': 144, 'C6': 144}
    transportation_cost = {'S1': {'C1': 0.08, 'C2': 52.33, 'C3': 73.57, 'C4': 1237.33, 'C5': 0.07, 'C6': 112.16}, 'S2': {'C1': 46.02, 'C2': 175.23, 'C3': 2026.83, 'C4': 299.89, 'C5': 966.53, 'C6': 1590.42}, 'S3': {'C1': 1031.74, 'C2': 78.13, 'C3': 99.02, 'C4': 277.07, 'C5': 884.45, 'C6': 1800.86}, 'S4': {'C1': 868.75, 'C2': 94.2, 'C3': 1776.34, 'C4': 285.48, 'C5': 868.85, 'C6': 86.55}, 'S5': {'C1': 1577, 'C2': 760.15, 'C3': 2090.19, 'C4': 43.2, 'C5': 1577.12, 'C6': 1095.17}, 'S6': {'C1': 49.14, 'C2': 4.33, 'C3': 2079.57, 'C4': 277.04, 'C5': 1032.01, 'C6': 1543.49}}
    if set(fixed_cost.keys()) != set(I):
        raise ValueError('fixed_cost keys do not match supplier set I')
    if set(demand.keys()) != set(J):
        raise ValueError('demand keys do not match store set J')
    if set(transportation_cost.keys()) != set(I):
        raise ValueError('transportation_cost keys do not match supplier set I')
    for i in I:
        if set(transportation_cost[i].keys()) != set(J):
            raise ValueError(f'transportation_cost[{i}] keys do not match store set J')
    total_demand = sum((demand[j] for j in J))
    M = {i: total_demand for i in I}
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(I, J, vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gp.quicksum((fixed_cost[i] * y[i] for i in I)) + gp.quicksum((transportation_cost[i][j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x[i, j] for i in I)) == demand[j], name='demand_' + j)
    for i in I:
        m.addConstr(gp.quicksum((x[i, j] for j in J)) <= M[i] * y[i], name='link_' + i)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_adidas_supplier_problem()