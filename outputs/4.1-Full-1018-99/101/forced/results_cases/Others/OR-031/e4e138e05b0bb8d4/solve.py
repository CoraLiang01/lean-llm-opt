import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Full_Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Full_Product_Name'] = df['Full_Product_Name'].str.strip()
products = df['Full_Product_Name'].tolist()

def to_float_series(s, colname):
    try:
        return s.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.") from e

def to_int_series(s, colname):
    try:
        return s.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-integer values.") from e
revenue = to_float_series(df['Revenue'], 'Revenue')
demand = to_int_series(df['Demand'], 'Demand')
init_inventory = to_int_series(df['Initial Inventory'], 'Initial Inventory')
revenue_dict = dict(zip(products, revenue))
demand_dict = dict(zip(products, demand))
inventory_dict = dict(zip(products, init_inventory))
if not set(revenue_dict.keys()) == set(demand_dict.keys()) == set(inventory_dict.keys()) == set(products):
    raise ValueError('Mismatch in product keys among parameters.')
m = gp.Model('DairyGoodsOrderFulfillment')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
demand_constrs = m.addConstrs((x_vars[i] <= demand_dict[i] for i in products), name='')
inventory_constrs = m.addConstrs((x_vars[i] <= inventory_dict[i] for i in products), name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for i in products:
        print(f'{i}: Fulfilled {int(round(x_vars[i].X))} units (Demand: {demand_dict[i]}, Inventory: {inventory_dict[i]}, Revenue/unit: {revenue_dict[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')