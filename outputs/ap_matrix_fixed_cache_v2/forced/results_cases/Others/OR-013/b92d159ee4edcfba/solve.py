import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM4/OnlineSalesinUSA.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def contains_4u(val):
    return '4u' in val.strip().casefold()
is_4u = df['Product Name'].apply(contains_4u)
df_4u = df[is_4u].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df_4u.columns:
        raise KeyError(f"Required column '{col}' not found in filtered 4U product data.")
product_ids = df_4u['Product Name'].tolist()
try:
    revenue_dict = dict(zip(product_ids, df_4u['Revenue'].astype(float)))
    demand_dict = dict(zip(product_ids, df_4u['Demand'].astype(int)))
    inventory_dict = dict(zip(product_ids, df_4u['Initial Inventory'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting numeric columns: {e}')
upper_bound_dict = {pid: min(demand_dict[pid], inventory_dict[pid]) for pid in product_ids}
m = gp.Model('4U_Product_Fulfillment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, ub=[upper_bound_dict[pid] for pid in product_ids], name='')
m.setObjective(gp.quicksum((revenue_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
for pid in product_ids:
    m.addConstr(x_vars[pid] <= inventory_dict[pid], name=f'inventory_{pid}')
    m.addConstr(x_vars[pid] <= demand_dict[pid], name=f'demand_{pid}')
    m.addConstr(x_vars[pid] >= 0, name=f'nonneg_{pid}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_revenue = m.objVal
    print(f'Optimal total revenue: {total_revenue:.2f}')
    print('--- Fulfillment Plan for 4U Products ---')
    for pid in product_ids:
        fulfilled = int(round(x_vars[pid].X))
        if fulfilled > 0:
            print(f'Product: {pid} | Fulfilled: {fulfilled} | Revenue/unit: {revenue_dict[pid]:.2f} | Total Revenue: {fulfilled * revenue_dict[pid]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')