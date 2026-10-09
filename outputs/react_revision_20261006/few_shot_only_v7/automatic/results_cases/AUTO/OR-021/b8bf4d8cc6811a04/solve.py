import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df_grouped = df.groupby('Product Name', as_index=False).agg({'Revenue': lambda x: list(x), 'Demand': lambda x: list(x), 'Initial Inventory': lambda x: list(x)})

    def flatten_or_sum(lst):
        if len(lst) == 1:
            return lst[0]
        else:
            return str(sum((float(v) for v in lst if v != '')))
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        df_grouped[col] = df_grouped[col].apply(flatten_or_sum)
    items = df_grouped['Product Name'].tolist()
    revenue = {}
    demand = {}
    inventory = {}
    for (idx, row) in df_grouped.iterrows():
        key = row['Product Name']
        try:
            revenue[key] = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Invalid or missing Revenue for product '{key}'")
        try:
            demand[key] = int(float(row['Demand']))
        except Exception:
            raise ValueError(f"Invalid or missing Demand for product '{key}'")
        try:
            inventory[key] = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f"Invalid or missing Initial Inventory for product '{key}'")
    for key in items:
        if key not in revenue or key not in demand or key not in inventory:
            raise ValueError(f"Missing coefficients for product '{key}'")
    m = gp.Model('Ecommerce_Summer_Clothes')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()