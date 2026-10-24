import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.set_index('Unnamed: 0', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    node_names = ['Depot', 'A', 'B', 'C']
    node_index = {name: idx for (idx, name) in enumerate(node_names)}
    N = list(range(len(node_names)))
    C = [node_index[name] for name in node_names[1:]]
    d = {}
    for i_name in node_names:
        for j_name in node_names:
            val = df.loc[i_name, j_name]
            if pd.isna(val):
                raise ValueError(f'Missing distance value for pair ({i_name}, {j_name})')
            d[node_index[i_name], node_index[j_name]] = float(val)
    m = gp.Model('TSP')
    m.setParam('MIPGap', 1e-06)
    x_keys = [(i, j) for i in N for j in N if i != j]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    u = m.addVars(C, vtype=GRB.CONTINUOUS, lb=1, ub=len(C), name='')
    m.setObjective(gp.quicksum((d[i, j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
    m.addConstr(gp.quicksum((x[0, j] for j in C)) == 1, name='depot_depart')
    m.addConstr(gp.quicksum((x[i, 0] for i in C)) == 1, name='depot_arrive')
    for i in C:
        m.addConstr(gp.quicksum((x[i, j] for j in N if j != i)) == 1, name=f'out_{i}')
        m.addConstr(gp.quicksum((x[j, i] for j in N if j != i)) == 1, name=f'in_{i}')
    for i in C:
        for j in C:
            if i != j:
                m.addConstr(u[i] - u[j] + len(C) * x[i, j] <= len(C) - 1, name=f'sub_{i}_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()