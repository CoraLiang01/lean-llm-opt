import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
id_col = 'id_number'
selected_ids = df[id_col].str.strip().str.casefold() == 'id999'
df_id999 = df[selected_ids].copy()
if df_id999.shape[0] == 0:
    raise ValueError("No products with id_number == 'id999' found in the dataset.")
df_id999.set_index(id_col, inplace=True)
try:
    revenue_dict = df_id999['Revenue'].astype(float).to_dict()
    demand_dict = df_id999['Demand'].astype(int).to_dict()
    inventory_dict = df_id999['Initial Inventory'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f'Error converting numeric columns: {e}')
products = list(df_id999.index)
for pid in products:
    if pid not in revenue_dict or pid not in demand_dict or pid not in inventory_dict:
        raise ValueError(f'Missing parameter(s) for product {pid}.')
m = gp.Model('MaximizeRevenue_id999')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
for pid in products:
    m.addConstr(x_vars[pid] <= inventory_dict[pid], name=f'inv_{pid}')
for pid in products:
    m.addConstr(x_vars[pid] <= demand_dict[pid], name=f'demand_{pid}')
m.setObjective(gp.quicksum((revenue_dict[pid] * x_vars[pid] for pid in products)), gp.GRB.MAXIMIZE)
m.optimize()