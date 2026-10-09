import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
product_name_col = 'Product Name'
s700_mask = df[product_name_col].str.strip().str.casefold().str.startswith('s700_')
df_s700 = df[s700_mask].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df_s700.columns:
        raise KeyError(f"Required column '{col}' not found in filtered data.")
df_s700.set_index('Product Name', inplace=True)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    try:
        df_s700[col] = pd.to_numeric(df_s700[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{col}' contains non-numeric values: {e}")
products = list(df_s700.index)
revenue = df_s700['Revenue'].to_dict()
demand = df_s700['Demand'].to_dict()
inventory = df_s700['Initial Inventory'].to_dict()
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f"Missing parameter for product '{i}'.")
m = gp.Model('S700_Fulfillment')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in products:
    m.addConstr(x_vars[i] <= int(demand[i]), name=f'demand_{i}')
    m.addConstr(x_vars[i] <= int(inventory[i]), name=f'inventory_{i}')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), gp.GRB.MAXIMIZE)
m.optimize()