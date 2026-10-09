import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode CSV at {csv_path} with tried encodings.')
    required_cols = ['Full_Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for col in required_cols:
        df[col] = df[col].astype(str).str.strip()
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    grouped = df.groupby('Full_Product_Name', sort=False, as_index=False).agg({'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    products = list(grouped['Full_Product_Name'])
    revenue = dict(zip(grouped['Full_Product_Name'], grouped['Revenue']))
    demand = dict(zip(grouped['Full_Product_Name'], grouped['Demand']))
    inventory = dict(zip(grouped['Full_Product_Name'], grouped['Initial Inventory']))
    for i in products:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product: {i}')
    var_ub = {i: min(demand[i], inventory[i]) for i in products}
    m = gp.Model('DairyGoodsRevenueMax')
    quantity_vars = m.addVars(products, lb=0, ub=[var_ub[i] for i in products], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in products), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in products), name='')
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