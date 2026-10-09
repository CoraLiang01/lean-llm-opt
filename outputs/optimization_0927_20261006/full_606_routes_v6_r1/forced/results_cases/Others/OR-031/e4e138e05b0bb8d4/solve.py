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

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to float: {e}")

def to_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to int: {e}")
revenue = dict(zip(df['Full_Product_Name'], to_float(df['Revenue'], 'Revenue')))
demand = dict(zip(df['Full_Product_Name'], to_int(df['Demand'], 'Demand')))
init_inventory = dict(zip(df['Full_Product_Name'], to_int(df['Initial Inventory'], 'Initial Inventory')))
for p in products:
    if p not in revenue or p not in demand or p not in init_inventory:
        raise KeyError(f"Missing parameter for product '{p}'.")
m = gp.Model('DairyGoodsOrderFulfillment')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= init_inventory[p] for p in products), name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for p in products:
        print(f'{p}: Fulfilled {int(round(x_vars[p].X))} units (Demand: {demand[p]}, Inventory: {init_inventory[p]}, Revenue/unit: {revenue[p]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')