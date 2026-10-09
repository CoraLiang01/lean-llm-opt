import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == encodings[-1]:
                raise
            continue
    mask = df['Product Name'].str.casefold() == 'fdk57'
    df_fdk57 = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_fdk57.columns:
            raise ValueError(f'Missing required column: {col}')
    if df_fdk57.empty:
        raise ValueError("No records found for Product Name 'FDK57'.")
    I = list(df_fdk57.index)
    A = {}
    d = {}
    s = {}
    for i in I:
        row = df_fdk57.loc[i]
        try:
            A[i] = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Non-numeric or missing Revenue for row {i}: {row['Revenue']}")
        try:
            d[i] = int(float(row['Demand']))
        except Exception:
            raise ValueError(f"Non-integer or missing Demand for row {i}: {row['Demand']}")
        try:
            s[i] = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f"Non-integer or missing Initial Inventory for row {i}: {row['Initial Inventory']}")
    if not set(A) == set(d) == set(s) == set(I):
        raise ValueError('Parameter keys do not match index set I.')
    m = gp.Model('FDK57_Car_Sales_Optimization')
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((A[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= d[i] for i in I), name='')
    m.addConstrs((quantity_vars[i] <= s[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')