import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for col in required_cols:
        df[col] = df[col].astype(str).str.strip()
    items = df['Product Name'].unique().tolist()
    revenue = {}
    demand = {}
    inventory = {}
    for i in items:
        mask = df['Product Name'].str.casefold() == i.casefold()
        try:
            revenue[i] = df.loc[mask, 'Revenue'].astype(float).sum()
        except Exception:
            raise ValueError(f"Non-numeric or missing Revenue for product '{i}'")
        try:
            demand[i] = int(df.loc[mask, 'Demand'].astype(float).sum())
        except Exception:
            raise ValueError(f"Non-numeric or missing Demand for product '{i}'")
        try:
            inventory[i] = int(df.loc[mask, 'Initial Inventory'].astype(float).sum())
        except Exception:
            raise ValueError(f"Non-numeric or missing Initial Inventory for product '{i}'")
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f"Missing parameter for product '{i}'")
    upper_bounds = {i: min(inventory[i], demand[i]) for i in items}
    m = gp.Model('Pizza_Fulfillment')
    quantity_vars = m.addVars(items, lb=0, ub=[upper_bounds[i] for i in items], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
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