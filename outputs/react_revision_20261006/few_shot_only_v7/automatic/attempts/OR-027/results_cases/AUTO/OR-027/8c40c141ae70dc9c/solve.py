import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == encodings[-1]:
                raise
            continue
    if 'Sub Category' not in df.columns:
        raise ValueError("Missing required column: 'Sub Category'")
    organ_mask = df['Sub Category'].str.casefold() == 'organ'
    organ_df = df[organ_mask].copy()
    if organ_df.empty:
        raise ValueError("No products with Sub Category 'Organ' found.")
    required_cols = ['Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in organ_df.columns:
            raise ValueError(f"Missing required column: '{col}'")
    organ_df = organ_df.reset_index(drop=False)
    items = organ_df['index'].tolist()

    def parse_numeric(series, name):
        vals = {}
        for (idx, val) in zip(organ_df['index'], organ_df[name]):
            if val == '':
                raise ValueError(f"Missing value in column '{name}' for product index {idx}")
            try:
                vals[idx] = float(val)
            except Exception:
                raise ValueError(f"Non-numeric value '{val}' in column '{name}' for product index {idx}")
        return vals
    revenue = parse_numeric(organ_df['Revenue'], 'Revenue')
    demand = parse_numeric(organ_df['Demand'], 'Demand')
    inventory = parse_numeric(organ_df['Initial Inventory'], 'Initial Inventory')
    for idx in items:
        if idx not in revenue or idx not in demand or idx not in inventory:
            raise ValueError(f'Missing coefficients for product index {idx}')
    m = gp.Model('Supermart_Organ_Revenue')
    m.setParam('MIPGap', 0.0001)
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