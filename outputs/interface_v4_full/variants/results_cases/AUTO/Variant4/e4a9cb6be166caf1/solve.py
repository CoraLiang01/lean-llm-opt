import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    centers = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8']
    districts = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    opening_cost = {'SC1': 12, 'SC2': 15, 'SC3': 18, 'SC4': 10, 'SC5': 14, 'SC6': 13, 'SC7': 16, 'SC8': 11}
    coverage = {'D1': {'SC1': 1, 'SC2': 0, 'SC3': 0, 'SC4': 0, 'SC5': 0, 'SC6': 0, 'SC7': 1, 'SC8': 0}, 'D2': {'SC1': 1, 'SC2': 1, 'SC3': 0, 'SC4': 0, 'SC5': 0, 'SC6': 0, 'SC7': 0, 'SC8': 0}, 'D3': {'SC1': 0, 'SC2': 1, 'SC3': 0, 'SC4': 0, 'SC5': 0, 'SC6': 0, 'SC7': 0, 'SC8': 1}, 'D4': {'SC1': 1, 'SC2': 0, 'SC3': 1, 'SC4': 0, 'SC5': 0, 'SC6': 0, 'SC7': 0, 'SC8': 1}, 'D5': {'SC1': 0, 'SC2': 1, 'SC3': 1, 'SC4': 0, 'SC5': 0, 'SC6': 0, 'SC7': 0, 'SC8': 0}, 'D6': {'SC1': 0, 'SC2': 0, 'SC3': 1, 'SC4': 1, 'SC5': 0, 'SC6': 0, 'SC7': 0, 'SC8': 0}, 'D7': {'SC1': 0, 'SC2': 0, 'SC3': 0, 'SC4': 1, 'SC5': 1, 'SC6': 0, 'SC7': 0, 'SC8': 0}, 'D8': {'SC1': 0, 'SC2': 0, 'SC3': 0, 'SC4': 0, 'SC5': 1, 'SC6': 1, 'SC7': 0, 'SC8': 1}, 'D9': {'SC1': 0, 'SC2': 0, 'SC3': 0, 'SC4': 0, 'SC5': 0, 'SC6': 1, 'SC7': 1, 'SC8': 0}, 'D10': {'SC1': 0, 'SC2': 0, 'SC3': 0, 'SC4': 0, 'SC5': 1, 'SC6': 0, 'SC7': 1, 'SC8': 0}}
    for d in districts:
        if d not in coverage:
            raise ValueError(f'Missing coverage data for district {d}')
        for c in centers:
            if c not in coverage[d]:
                raise ValueError(f'Missing coverage data for center {c} in district {d}')
    m = gp.Model('ServiceCenterSetCover')
    m.Params.MIPGap = 0.0001
    y = m.addVars(centers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_cost[i] * y[i] for i in centers)), GRB.MINIMIZE)
    for j in districts:
        m.addConstr(gp.quicksum((coverage[j][i] * y[i] for i in centers)) >= 1, name=f'cov_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()