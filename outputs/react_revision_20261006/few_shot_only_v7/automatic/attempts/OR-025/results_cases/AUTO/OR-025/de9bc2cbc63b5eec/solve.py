import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with any of the specified encodings.')
    mask = df['Product Name'].str.casefold().str.contains('tablet')
    df_tablet = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_tablet.columns:
            raise ValueError(f'Missing required column: {col}')
    df_tablet_grouped = df_tablet.groupby('Product Name', as_index=False).agg({'Revenue': 'first', 'Demand': lambda x: sum(pd.to_numeric(x, errors='raise')), 'Initial Inventory': lambda x: sum(pd.to_numeric(x, errors='raise'))})
    items = df_tablet_grouped['Product Name'].tolist()
    if not items:
        raise ValueError("No eligible 'TABLET' products found in the source data.")
    try:
        revenue = dict(zip(items, pd.to_numeric(df_tablet_grouped['Revenue'], errors='raise')))
        demand = dict(zip(items, pd.to_numeric(df_tablet_grouped['Demand'], errors='raise')))
        inventory = dict(zip(items, pd.to_numeric(df_tablet_grouped['Initial Inventory'], errors='raise')))
    except Exception as e:
        raise ValueError(f'Error converting parameters to numeric: {e}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing data for product: {i}')
    m = gp.Model('SmartphoneRetailOutlet_TABLET')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')