import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for encoding in decode_errors:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Unable to decode CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df[df['Product Name'].str.strip() != '']
    product_names = df['Product Name'].unique().tolist()
    revenue_dict = {}
    demand_dict = {}
    inventory_dict = {}
    for name in product_names:
        mask = df['Product Name'] == name
        try:
            revenue = pd.to_numeric(df.loc[mask, 'Revenue'], errors='coerce').fillna(0).sum()
            demand = pd.to_numeric(df.loc[mask, 'Demand'], errors='coerce').fillna(0).sum()
            inventory = pd.to_numeric(df.loc[mask, 'Initial Inventory'], errors='coerce').fillna(0).sum()
        except Exception as e:
            raise ValueError(f"Error converting numeric fields for product '{name}': {e}")
        revenue_dict[name] = float(revenue)
        demand_dict[name] = int(demand)
        inventory_dict[name] = int(inventory)
    for name in product_names:
        if name not in revenue_dict or name not in demand_dict or name not in inventory_dict:
            raise ValueError(f"Missing parameter(s) for product '{name}'")
        if not isinstance(revenue_dict[name], (int, float)):
            raise ValueError(f"Non-numeric revenue for product '{name}'")
        if not isinstance(demand_dict[name], int):
            raise ValueError(f"Non-integer demand for product '{name}'")
        if not isinstance(inventory_dict[name], int):
            raise ValueError(f"Non-integer inventory for product '{name}'")
    m = gp.Model('WomenClothingEcommerceSales')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue_dict[i] * quantity_vars[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand_dict[i] for i in product_names), name='')
    m.addConstrs((quantity_vars[i] <= inventory_dict[i] for i in product_names), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')