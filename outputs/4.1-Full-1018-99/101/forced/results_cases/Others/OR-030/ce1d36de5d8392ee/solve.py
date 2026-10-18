import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
product_name_col = 'Product Name'
fdk57_mask = df[product_name_col].str.strip().str.casefold() == 'fdk57'
df_fdk57 = df[fdk57_mask].copy()
if df_fdk57.empty:
    raise ValueError("No rows found with 'Product Name' exactly equal to 'FDK57'.")
fdk57_keys = df_fdk57.index.tolist()

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")

def to_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")
revenue_dict = to_float(df_fdk57['Revenue'], 'Revenue').to_dict()
demand_dict = to_int(df_fdk57['Demand'], 'Demand').to_dict()
inventory_dict = to_int(df_fdk57['Initial Inventory'], 'Initial Inventory').to_dict()
for k in fdk57_keys:
    if k not in revenue_dict or k not in demand_dict or k not in inventory_dict:
        raise ValueError(f'Missing parameter(s) for FDK57 model at row index {k}.')
m = gp.Model('FDK57_Fulfillment')
x_vars = m.addVars(fdk57_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue_dict[k] * x_vars[k] for k in fdk57_keys)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[k] <= demand_dict[k] for k in fdk57_keys), name='')
m.addConstrs((x_vars[k] <= inventory_dict[k] for k in fdk57_keys), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan for FDK57 Models ---')
    for k in fdk57_keys:
        fulfilled = int(round(x_vars[k].X))
        print(f'Row {k}: Fulfill {fulfilled} units (Demand: {demand_dict[k]}, Inventory: {inventory_dict[k]}, Revenue/unit: {revenue_dict[k]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')