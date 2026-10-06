import gurobipy as gp
from gurobipy import GRB

def solve_brewco_transportation():
    S = ['S1', 'S2', 'S3', 'S4']
    C = ['C1', 'C2', 'C3', 'C4']
    customer_demand = {'C1': 94, 'C2': 39, 'C3': 65, 'C4': 435}
    supply_capacity = {'S1': 2531, 'S2': 20, 'S3': 210, 'S4': 241}
    transportation_costs = {'S1': {'C1': 543.756480860856, 'C2': 23.685276141764653, 'C3': 23.676386730773032, 'C4': 447.75143678673766}, 'S2': {'C1': 883.9151090405642, 'C2': 0.04977684765576961, 'C3': 0.0350986687216299, 'C4': 44.45588531711622}, 'S3': {'C1': 537.3456896658107, 'C2': 23.769274659075112, 'C3': 498.95659249465467, 'C4': 440.60737890439776}, 'S4': {'C1': 1791.493192397229, 'C2': 68.21633865655126, 'C3': 1432.4837339656747, 'C4': 1527.7635425462734}}
    if set(customer_demand.keys()) != set(C):
        raise ValueError('Customer demand keys do not match outlets set C.')
    if set(supply_capacity.keys()) != set(S):
        raise ValueError('Supply capacity keys do not match plants set S.')
    if set(transportation_costs.keys()) != set(S):
        raise ValueError('Transportation costs keys do not match plants set S.')
    for s in S:
        if set(transportation_costs[s].keys()) != set(C):
            raise ValueError(f'Transportation costs for plant {s} do not match outlets set C.')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(S, C, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.LinExpr()
    for s in S:
        for c in C:
            obj += transportation_costs[s][c] * x[s, c]
    m.setObjective(obj, GRB.MINIMIZE)
    for c in C:
        m.addConstr(gp.quicksum((x[s, c] for s in S)) == customer_demand[c], name='d_' + c)
    for s in S:
        m.addConstr(gp.quicksum((x[s, c] for c in C)) <= supply_capacity[s], name='s_' + s)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for s in S:
            for c in C:
                v = x[s, c]
                print(v.VarName, v.X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_brewco_transportation()