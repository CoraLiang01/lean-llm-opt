import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    bi_path = 'Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
    df = pd.read_csv(bi_path, dtype=str, keep_default_na=False)
    bi_row = df.loc[df['Process'] == 'BI']
    if bi_row.empty:
        raise ValueError('BI row not found in CSV')
    bi_row = bi_row.iloc[0]
    task_cols = df.columns[1:]
    if len(task_cols) < 40:
        raise ValueError('Not enough task columns in CSV')
    BI_dict = {}
    for col in task_cols[:40]:
        try:
            BI_dict[int(col)] = float(bi_row[col])
        except ValueError as e:
            raise ValueError(f'Invalid BI value for task {col}') from e
    missing_tasks = [j for j in range(1, 41) if j not in BI_dict]
    if missing_tasks:
        raise ValueError(f'Missing BI values for tasks: {missing_tasks}')
    f = {1: 1.33, 2: 2.0, 3: 2.66}
    p = {}
    for i in f:
        for j in range(1, 41):
            p[i, j] = BI_dict[j] / f[i]
    I = [1, 2, 3]
    J = list(range(1, 41))
    m = gurobipy.Model('CPU_Task_Assignment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.BINARY, name='')
    T = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='T')
    m.setObjective(T, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) == 1 for j in J), name='')
    m.addConstrs((gurobipy.quicksum((p[i, j] * x[i, j] for j in J)) <= T for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'Optimal objective value: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.Status}')