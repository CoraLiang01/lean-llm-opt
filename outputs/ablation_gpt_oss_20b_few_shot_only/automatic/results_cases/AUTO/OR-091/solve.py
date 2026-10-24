import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df['credits'] = pd.to_numeric(df['credits'], errors='coerce')
    df['interest_points'] = pd.to_numeric(df['interest_points'], errors='coerce')
    or_rows = df[df['discipline'] == 'Operations Research']
    indices = or_rows.index.tolist()
    c = {i: int(df.at[i, 'credits']) for i in indices}
    p = {i: int(df.at[i, 'interest_points']) for i in indices}
    for i in indices:
        if i not in c:
            raise ValueError(f'Missing credits for row {i}')
        if i not in p:
            raise ValueError(f'Missing interest_points for row {i}')
    m = gurobipy.Model('OR_Course_Selection')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(indices, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((p[i] * x[i] for i in indices)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((c[i] * x[i] for i in indices)) <= 20, name='CreditLimit')
    m.optimize()
    return m
m = solve_problem()