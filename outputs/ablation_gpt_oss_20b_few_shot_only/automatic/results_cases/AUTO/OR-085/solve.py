import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    nodes = list(range(1, 16))
    d = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            val_str = df.loc[df['Unnamed: 0'] == str(i), str(j)].values[0]
            if val_str == '':
                val_str = df.loc[df['Unnamed: 0'] == str(j), str(i)].values[0]
            if val_str == '':
                raise ValueError(f'Missing distance for pair {i},{j}')
            d[i, j] = float(val_str)
    for i in nodes:
        for j in nodes:
            if i != j and (i, j) not in d:
                raise ValueError(f'Distance missing for pair {i},{j}')
    m = gurobipy.Model('TSP_MTZ')
    m.setParam('MIPGap', 1e-06)
    x_keys = [(i, j) for i in nodes for j in nodes if i != j]
    x = m.addVars(x_keys, vtype=GRB.BINARY, name='')
    u_keys = [i for i in nodes if i != 1]
    u = m.addVars(u_keys, lb=2, ub=15, vtype=GRB.CONTINUOUS, name='')
    for i in nodes:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'out_{i}')
        m.addConstr(gurobipy.quicksum((x[j, i] for j in nodes if j != i)) == 1, name=f'in_{i}')
    for i in u_keys:
        for j in u_keys:
            if i != j:
                m.addConstr(u[i] - u[j] + 15 * x[i, j] <= 14, name=f'sub_{i}_{j}')
    m.setObjective(gurobipy.quicksum((d[i, j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()