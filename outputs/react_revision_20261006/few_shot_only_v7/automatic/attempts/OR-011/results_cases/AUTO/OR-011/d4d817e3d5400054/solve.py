import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == tried_encodings[-1]:
                raise
            continue
    id_col = 'id_number'
    id999_mask = df[id_col].str.casefold() == 'id999'
    df_id999 = df[id999_mask].copy()
    required_cols = ['id_number', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_id999.columns:
            raise ValueError(f'Missing required column: {col}')
    index_keys = list(df_id999.index)
    product_info = {k: df_id999.loc[k, :].to_dict() for k in index_keys}
    revenue = {}
    demand = {}
    inventory = {}
    for k in index_keys:
        row = product_info[k]
        try:
            revenue[k] = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Invalid or missing Revenue for row {k}: {row['Revenue']}")
        try:
            demand[k] = int(float(row['Demand']))
        except Exception:
            raise ValueError(f"Invalid or missing Demand for row {k}: {row['Demand']}")
        try:
            inventory[k] = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f"Invalid or missing Initial Inventory for row {k}: {row['Initial Inventory']}")
    if not set(revenue) == set(demand) == set(inventory) == set(index_keys):
        raise ValueError('Mismatch in index sets for coefficients.')
    m = gp.Model('Supermarket_id999_Allocation')
    quantity_vars = m.addVars(index_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[k] * quantity_vars[k] for k in index_keys)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[k] <= inventory[k] for k in index_keys), name='')
    m.addConstrs((quantity_vars[k] <= demand[k] for k in index_keys), name='')
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