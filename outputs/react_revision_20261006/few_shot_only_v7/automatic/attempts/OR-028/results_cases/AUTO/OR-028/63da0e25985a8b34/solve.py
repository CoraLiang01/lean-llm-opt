import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
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
    df = df.drop_duplicates(subset=required_cols)
    products = df['Product Name'].tolist()
    if len(set(products)) != len(products):
        raise ValueError('Duplicate product names detected; please ensure unique identifiers.')
    try:
        revenue = pd.to_numeric(df['Revenue'], errors='raise')
        demand = pd.to_numeric(df['Demand'], errors='raise')
        inventory = pd.to_numeric(df['Initial Inventory'], errors='raise')
    except Exception as e:
        raise ValueError(f'Error converting numeric columns: {e}')
    revenue_dict = dict(zip(products, revenue))
    demand_dict = dict(zip(products, demand))
    inventory_dict = dict(zip(products, inventory))
    for p in products:
        if p not in revenue_dict or p not in demand_dict or p not in inventory_dict:
            raise ValueError(f'Missing coefficients for product: {p}')
    m = gp.Model('WomenClothingEcommerceSales')
    quantity_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue_dict[p] * quantity_vars[p] for p in products)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[p] <= demand_dict[p] for p in products), name='')
    m.addConstrs((quantity_vars[p] <= inventory_dict[p] for p in products), name='')
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