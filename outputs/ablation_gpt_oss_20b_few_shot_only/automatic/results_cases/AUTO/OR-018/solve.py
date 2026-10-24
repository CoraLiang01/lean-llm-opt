import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns: {missing_cols}')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
    df['Demand'] = pd.to_numeric(df['Demand'], errors='raise')
    df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise')
    df_baby = df[df['Product Name'].str.startswith('Baby')]
    if df_baby.empty:
        raise ValueError("No products with names starting with 'Baby' found.")
    product_names = df_baby['Product Name'].tolist()
    if len(product_names) != len(set(product_names)):
        raise ValueError('Duplicate product names found among Baby products.')
    r = dict(zip(product_names, df_baby['Revenue']))
    d = dict(zip(product_names, df_baby['Demand']))
    I_dict = dict(zip(product_names, df_baby['Initial Inventory']))
    for p in product_names:
        if p not in r or p not in d or p not in I_dict:
            raise ValueError(f'Missing parameter for product {p}')
    m = gurobipy.Model('Baby_Product_Revenue_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=1, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((r[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
    m.addConstrs((x[p] <= d[p] for p in product_names), name='')
    m.addConstrs((x[p] <= I_dict[p] for p in product_names), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')