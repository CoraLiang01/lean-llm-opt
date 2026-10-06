import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.dropna(subset=required_cols)
    I = df['Product Name'].astype(str).tolist()
    if len(set(I)) != len(I):
        raise ValueError('Duplicate Product Name entries found; ensure unique identifiers.')
    try:
        A = dict(zip(I, df['Revenue'].astype(float)))
        d = dict(zip(I, df['Demand'].astype(float)))
        I_inv = dict(zip(I, df['Initial Inventory'].astype(float)))
    except Exception as e:
        raise ValueError(f'Error converting parameter columns to float: {e}')
    for i in I:
        if i not in A or i not in d or i not in I_inv:
            raise ValueError(f'Missing parameter for product {i}')
    m = gp.Model('FrenchBakery_RevMax')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((A[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.addConstrs((x[i] <= I_inv[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()