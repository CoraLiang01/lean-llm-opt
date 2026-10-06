import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    nodes = ['Depot', 'A', 'B', 'C']
    customers = ['A', 'B', 'C']
    d = {'Depot': {'Depot': 0, 'A': 28, 'B': 41, 'C': 63}, 'A': {'Depot': 28, 'A': 0, 'B': 27, 'C': 87}, 'B': {'Depot': 41, 'A': 27, 'B': 0, 'C': 81}, 'C': {'Depot': 63, 'A': 87, 'B': 81, 'C': 0}}
    for i in nodes:
        for j in nodes:
            if i != j and j not in d[i]:
                raise ValueError(f'Missing distance from {i} to {j}')
    m = gp.Model('Courier_TSP')
    x = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=GRB.BINARY, name='')
    u = m.addVars(customers, vtype=GRB.INTEGER, lb=1, ub=3, name='')
    m.setObjective(gp.quicksum((d[i][j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
    for i in nodes:
        m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'leave_{i}')
    for j in nodes:
        m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'enter_{j}')
    for i in customers:
        for j in customers:
            if i != j:
                m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()