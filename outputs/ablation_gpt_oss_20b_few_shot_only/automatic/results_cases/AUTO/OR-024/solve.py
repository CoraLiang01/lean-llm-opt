import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV.")
        df[col] = pd.to_numeric(df[col], errors='raise')
    if 'Product Name' not in df.columns:
        raise ValueError("Missing required column 'Product Name' in CSV.")
    df_filtered = df[df['Product Name'].str.startswith('S700_')].copy()
    I = df_filtered['Product Name'].tolist()
    R_i = dict(zip(I, df_filtered['Revenue']))
    D_i = dict(zip(I, df_filtered['Demand']))
    I_i = dict(zip(I, df_filtered['Initial Inventory']))
    for prod in I:
        if prod not in R_i or prod not in D_i or prod not in I_i:
            raise ValueError(f"Missing parameter data for product '{prod}'.")
    m = gurobipy.Model('S700_Revenue_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((R_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= D_i[i] for i in I), name='')
    m.addConstrs((x[i] <= I_i[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()