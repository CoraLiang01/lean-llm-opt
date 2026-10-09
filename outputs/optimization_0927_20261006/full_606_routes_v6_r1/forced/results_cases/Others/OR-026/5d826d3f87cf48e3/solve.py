import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def is_fashion(row):
    return 'fashion' in row['Product Name'].strip().casefold()
fashion_mask = df.apply(is_fashion, axis=1)
fashion_df = df[fashion_mask].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in fashion_df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    mask = fashion_df[col].apply(lambda x: x.strip() != '' and re.match('^-?\\d+(\\.\\d+)?$', x.strip()))
    fashion_df = fashion_df[mask]
fashion_df['Revenue'] = fashion_df['Revenue'].astype(float)
fashion_df['Demand'] = fashion_df['Demand'].astype(float)
fashion_df['Initial Inventory'] = fashion_df['Initial Inventory'].astype(float)
fashion_df = fashion_df.set_index('Product Name', drop=False)
fashion_products = list(fashion_df.index)
revenue = fashion_df['Revenue'].to_dict()
demand = fashion_df['Demand'].to_dict()
inventory = fashion_df['Initial Inventory'].to_dict()
m = gp.Model('FashionRevenueMaximization')
x_vars = m.addVars(fashion_products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for i in fashion_products:
    m.addConstr(x_vars[i] <= demand[i], name=f'demand_{i}')
    m.addConstr(x_vars[i] <= inventory[i], name=f'inventory_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'nonneg_{i}')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in fashion_products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fashion Product Fulfillment Plan ---')
    for i in fashion_products:
        fulfilled = x_vars[i].X
        if fulfilled > 1e-06:
            print(f'{i}: Fulfill {fulfilled:.2f} units (Revenue per unit: {revenue[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')