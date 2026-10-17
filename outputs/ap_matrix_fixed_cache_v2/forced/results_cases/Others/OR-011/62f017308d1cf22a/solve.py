import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df_id999 = df[df['id_number'].str.strip().str.casefold() == 'id999']
if df_id999.shape[0] == 0:
    raise ValueError("No products with id_number == 'id999' found in the dataset.")
product_ids = df_id999['id_number'].tolist()

def safe_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")

def safe_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")
revenue_dict = dict(zip(df_id999['id_number'], safe_float(df_id999['Revenue'], 'Revenue')))
demand_dict = dict(zip(df_id999['id_number'], safe_int(df_id999['Demand'], 'Demand')))
inventory_dict = dict(zip(df_id999['id_number'], safe_int(df_id999['Initial Inventory'], 'Initial Inventory')))
for pid in product_ids:
    if pid not in revenue_dict or pid not in demand_dict or pid not in inventory_dict:
        raise ValueError(f'Missing parameter(s) for product {pid}.')
m = gp.Model('MaximizeRevenue_id999')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory_dict[i] for i in product_ids), name='')
m.addConstrs((x_vars[i] <= demand_dict[i] for i in product_ids), name='')
m.optimize()