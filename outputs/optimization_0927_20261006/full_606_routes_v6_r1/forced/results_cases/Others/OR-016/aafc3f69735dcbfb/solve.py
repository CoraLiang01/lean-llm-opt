import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
product_ids = df['Product Name'].tolist()

def to_int_series(series, colname):
    try:
        return series.astype(str).str.strip().astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be fully converted to int: {e}")
revenue_dict = dict(zip(product_ids, to_int_series(df['Revenue'], 'Revenue')))
demand_dict = dict(zip(product_ids, to_int_series(df['Demand'], 'Demand')))
inventory_dict = dict(zip(product_ids, to_int_series(df['Initial Inventory'], 'Initial Inventory')))
m = gp.Model('RetailMerchandiseAllocation')
x_vars = m.addVars(product_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
for i in product_ids:
    m.addConstr(x_vars[i] <= demand_dict[i], name=f'demand_{i}')
    m.addConstr(x_vars[i] <= inventory_dict[i], name=f'inventory_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Optimal Fulfillment Plan ---')
    for i in product_ids:
        print(f'{i}: Fulfill {x_vars[i].X:.2f} units (Demand: {demand_dict[i]}, Inventory: {inventory_dict[i]}, Revenue/unit: {revenue_dict[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')