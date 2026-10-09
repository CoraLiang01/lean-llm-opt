import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
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
    df['Product Name'] = df['Product Name'].str.strip()
    items = df['Product Name'].tolist()
    if len(set(items)) != len(items):
        raise ValueError('Duplicate Product Name entries found; keys must be unique.')
    try:
        revenue = pd.to_numeric(df['Revenue'], errors='raise')
        demand = pd.to_numeric(df['Demand'], errors='raise')
        inventory = pd.to_numeric(df['Initial Inventory'], errors='raise')
    except Exception as e:
        raise ValueError(f'Error converting numeric columns: {e}')
    revenue_dict = dict(zip(items, revenue))
    demand_dict = dict(zip(items, demand))
    inventory_dict = dict(zip(items, inventory))
    for i in items:
        if i not in revenue_dict or i not in demand_dict or i not in inventory_dict:
            raise ValueError(f'Missing coefficients for product: {i}')
    m = gp.Model('Retail_Revenue_Allocation')
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue_dict[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand_dict[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory_dict[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')