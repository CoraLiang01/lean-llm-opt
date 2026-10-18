import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def is_eles(ref):
    return 'ele-s' in ref.strip().casefold()
eles_mask = df['Product_Reference'].apply(is_eles)
eles_df = df[eles_mask].copy()
if eles_df.empty:
    raise ValueError("No products found with 'ELE-S' in 'Product_Reference'.")
eles_df['Product_Reference'] = eles_df['Product_Reference'].apply(lambda x: x.strip())
eles_df = eles_df.set_index('Product_Reference', drop=False)
try:
    revenue = eles_df['Revenue'].astype(float)
    demand = eles_df['Demand'].astype(float)
    inventory = eles_df['Initial Inventory'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting numeric columns: {e}')
eles_products = list(eles_df.index)
if not set(eles_products) == set(revenue.index) == set(demand.index) == set(inventory.index):
    raise ValueError('Mismatch in product indices among parameters.')
m = gp.Model('ELE-S_Fulfillment')
x_vars = m.addVars(eles_products, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in eles_products:
    m.addConstr(x_vars[i] <= demand[i], name=f'demand_{i}')
    m.addConstr(x_vars[i] <= inventory[i], name=f'inventory_{i}')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in eles_products)), gp.GRB.MAXIMIZE)
m.optimize()