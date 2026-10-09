import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == tried_encodings[-1]:
                raise
            continue
    required_cols = ['Product Name', 'Labor per unit', 'Material per unit', 'Selling Price', 'Variable Cost']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df_grouped = df.groupby('Product Name', as_index=False).agg({'Labor per unit': 'sum', 'Material per unit': 'sum', 'Selling Price': 'sum', 'Variable Cost': 'sum'})
    products = df_grouped['Product Name'].tolist()
    try:
        labor = pd.to_numeric(df_grouped['Labor per unit'])
        material = pd.to_numeric(df_grouped['Material per unit'])
        selling_price = pd.to_numeric(df_grouped['Selling Price'])
        variable_cost = pd.to_numeric(df_grouped['Variable Cost'])
    except Exception as e:
        raise ValueError(f'Error converting coefficients to numeric: {e}')
    l_p = dict(zip(products, labor))
    m_p = dict(zip(products, material))
    s_p = dict(zip(products, selling_price))
    v_p = dict(zip(products, variable_cost))
    L = 1650
    M = 1850
    F = 4500
    m = gp.Model('RedBeanClothingFactory')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum(((s_p[p] - v_p[p]) * quantity_vars[p] for p in products)) - F, GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((l_p[p] * quantity_vars[p] for p in products)) <= L, name='labor')
    m.addConstr(gp.quicksum((m_p[p] * quantity_vars[p] for p in products)) <= M, name='material')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')