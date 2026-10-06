import gurobipy as gp
from gurobipy import GRB

def solve_supplier_assignment():
    F = ['S1', 'S2']
    C = ['C1', 'C2']
    fixed_cost = {'S1': 105.97, 'S2': 85.31}
    demand = {'C1': 144, 'C2': 216}
    transportation_cost = {('S1', 'C1'): 2358.39, ('S1', 'C2'): 1492.08, ('S2', 'C1'): 0.07, ('S2', 'C2'): 52.32}
    if set(fixed_cost.keys()) != set(F):
        raise ValueError('Fixed cost data missing or extra entries for suppliers.')
    if set(demand.keys()) != set(C):
        raise ValueError('Demand data missing or extra entries for customers.')
    if set(transportation_cost.keys()) != {(i, j) for i in F for j in C}:
        raise ValueError('Transportation cost data missing or extra entries for supplier-customer pairs.')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(F, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(F, C, vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gp.quicksum((fixed_cost[i] * y[i] for i in F)) + gp.quicksum((transportation_cost[i, j] * x[i, j] for i in F for j in C))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in C:
        m.addConstr(gp.quicksum((x[i, j] for i in F)) == demand[j], name='demand_' + j)
    total_demand = sum((demand[j] for j in C))
    for i in F:
        m.addConstr(gp.quicksum((x[i, j] for j in C)) <= total_demand * y[i], name='link_' + i)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_supplier_assignment()