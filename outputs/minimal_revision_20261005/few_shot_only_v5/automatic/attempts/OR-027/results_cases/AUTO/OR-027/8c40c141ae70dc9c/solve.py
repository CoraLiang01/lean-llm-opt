import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    if 'Sub Category' not in df.columns:
        raise ValueError("Missing required column: 'Sub Category'")
    organ_mask = df['Sub Category'].str.casefold() == 'organ'
    df_organ = df[organ_mask].copy()
    if df_organ.empty:
        raise ValueError("No 'Organ' products found in 'Sub Category'.")
    required_cols = ['Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_organ.columns:
            raise ValueError(f"Missing required column: '{col}'")
    items = list(df_organ.index)
    try:
        revenue = df_organ['Revenue'].to_dict()
        inventory = df_organ['Initial Inventory'].to_dict()
        demand = df_organ['Demand'].to_dict()
    except Exception as e:
        raise ValueError(f'Error extracting coefficients: {e}')
    for i in items:
        for (coeff_dict, name) in [(revenue, 'Revenue'), (inventory, 'Initial Inventory'), (demand, 'Demand')]:
            if i not in coeff_dict:
                raise ValueError(f'Missing {name} for item {i}')
            if pd.isnull(coeff_dict[i]):
                raise ValueError(f'Null {name} for item {i}')
            if not isinstance(coeff_dict[i], (int, float)):
                raise ValueError(f'Non-numeric {name} for item {i}: {coeff_dict[i]}')
    m = gp.Model('Supermart_Organ_Fulfillment')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()