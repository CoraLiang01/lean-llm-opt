import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with tried encodings.')
    mask = df['Product Name'].astype(str).str.casefold().str.startswith('tablet_')
    tablet_df = df.loc[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in tablet_df.columns:
            raise ValueError(f'Missing required column: {col}')
    tablet_df = tablet_df.dropna(subset=required_cols)
    items = tablet_df['Product Name'].astype(str).tolist()
    if len(items) == 0:
        raise ValueError("No eligible 'TABLET_' products found in the source data.")
    try:
        revenue = tablet_df.set_index('Product Name')['Revenue'].astype(float).to_dict()
        demand = tablet_df.set_index('Product Name')['Demand'].astype(float).to_dict()
        inventory = tablet_df.set_index('Product Name')['Initial Inventory'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error extracting coefficients: {e}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficients for product: {i}')
    m = gp.Model('Tablet_Revenue_Max')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()