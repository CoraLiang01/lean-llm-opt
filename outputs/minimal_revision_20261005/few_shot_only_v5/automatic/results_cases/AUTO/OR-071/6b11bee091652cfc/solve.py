import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Labor per unit', 'Material per unit', 'Selling Price', 'Variable Cost']
    if not all((col in df.columns for col in required_cols)):
        raise ValueError(f'Missing required columns in {csv_path}')
    df = df.groupby('Product Name', as_index=False).agg({'Labor per unit': 'sum', 'Material per unit': 'sum', 'Selling Price': 'first', 'Variable Cost': 'first'})
    products = df['Product Name'].tolist()
    labor = dict(zip(df['Product Name'], df['Labor per unit']))
    material = dict(zip(df['Product Name'], df['Material per unit']))
    selling_price = dict(zip(df['Product Name'], df['Selling Price']))
    variable_cost = dict(zip(df['Product Name'], df['Variable Cost']))
    L = 1650
    M = 1850
    F = 4500
    m = gp.Model('RedBeanClothingFactory')
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum(((selling_price[i] - variable_cost[i]) * x[i] for i in products)) - F, GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor[i] * x[i] for i in products)) <= L, name='labor')
    m.addConstr(gp.quicksum((material[i] * x[i] for i in products)) <= M, name='material')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()