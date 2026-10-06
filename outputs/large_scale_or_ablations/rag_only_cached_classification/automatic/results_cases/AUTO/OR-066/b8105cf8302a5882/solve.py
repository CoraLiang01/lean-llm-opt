import gurobipy as gp
from gurobipy import GRB

def solve_supplier_supermarket():
    suppliers = ['S1', 'S2']
    supermarkets = ['C1', 'C2']
    fixed_cost = {'S1': 105.97, 'S2': 85.31}
    demand = {'C1': 144, 'C2': 216}
    transportation_cost = {('S1', 'C1'): 2358.39, ('S1', 'C2'): 1492.08, ('S2', 'C1'): 0.07, ('S2', 'C2'): 52.32}
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    for j in supermarkets:
        if j not in demand:
            raise ValueError(f'Missing demand for supermarket {j}')
    for i in suppliers:
        for j in supermarkets:
            if (i, j) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({i},{j})')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x = m.addVars(suppliers, supermarkets, vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_cost[i, j] * x[i, j] for i in suppliers for j in supermarkets))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in supermarkets:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name='demand_' + j)
    for i in suppliers:
        for j in supermarkets:
            m.addConstr(x[i, j] <= demand[j] * y[i], name='link_%s_%s' % (i, j))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_supplier_supermarket()