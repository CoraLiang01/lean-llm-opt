import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with supported encodings.')
    if 'Product Name' not in df.columns:
        raise ValueError("Missing 'Product Name' column in source data.")
    s700_mask = df['Product Name'].str.contains('S700_', case=False, na=False)
    df_s700 = df[s700_mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_s700.columns:
            raise ValueError(f"Missing required column '{col}' in filtered data.")
    df_s700['Revenue'] = pd.to_numeric(df_s700['Revenue'], errors='raise')
    df_s700['Demand'] = pd.to_numeric(df_s700['Demand'], errors='raise')
    df_s700['Initial Inventory'] = pd.to_numeric(df_s700['Initial Inventory'], errors='raise')
    df_s700 = df_s700.groupby('Product Name', as_index=False).agg({'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_s700['Product Name'].tolist()
    revenue = dict(zip(df_s700['Product Name'], df_s700['Revenue']))
    demand = dict(zip(df_s700['Product Name'], df_s700['Demand']))
    inventory = dict(zip(df_s700['Product Name'], df_s700['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f"Missing coefficients for product '{i}'.")
    m = gp.Model('NRM_S700_Fulfillment')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')