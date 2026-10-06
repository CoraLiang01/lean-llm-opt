import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Labor per unit', 'Material per unit', 'Selling Price', 'Variable Cost']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.drop_duplicates(subset=['Product Name'], keep='first')
    I = df['Product Name'].tolist()
    labor = dict(zip(df['Product Name'], df['Labor per unit']))
    material = dict(zip(df['Product Name'], df['Material per unit']))
    selling_price = dict(zip(df['Product Name'], df['Selling Price']))
    variable_cost = dict(zip(df['Product Name'], df['Variable Cost']))
    L = 1650
    M = 1850
    F = 4500
    for i in I:
        if i not in labor or pd.isnull(labor[i]) or i not in material or pd.isnull(material[i]) or (i not in selling_price) or pd.isnull(selling_price[i]) or (i not in variable_cost) or pd.isnull(variable_cost[i]):
            raise ValueError(f'Missing coefficient(s) for product: {i}')
    m = gp.Model('RedBeanClothingFactory')
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum(((selling_price[i] - variable_cost[i]) * x[i] for i in I)) - F, GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor[i] * x[i] for i in I)) <= L, name='labor')
    m.addConstr(gp.quicksum((material[i] * x[i] for i in I)) <= M, name='material')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()