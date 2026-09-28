import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
df = pd.read_csv(csv_path, sep=',')

def is_aalop(row):
    return str(row['Product Name']).strip().casefold().startswith('aalop')
aalop_mask = df.apply(is_aalop, axis=1)
aalop_df = df[aalop_mask].copy()
if aalop_df.empty:
    raise ValueError("No products classified under 'Aalop' found in the data.")
aalop_products = list(aalop_df['Product Name'])
revenue = {}
demand = {}
init_inventory = {}
for idx, row in aalop_df.iterrows():
    pname = str(row['Product Name'])
    try:
        revenue[pname] = int(row['Revenue'])
        demand[pname] = int(row['Demand'])
        inv = row['Initial Inventory']
        if not (isinstance(inv, (int, float)) and (not pd.isnull(inv))):
            raise ValueError(f"Initial Inventory missing or invalid for product '{pname}'")
        init_inventory[pname] = int(math.floor(inv))
    except Exception as e:
        raise ValueError(f"Error parsing parameters for product '{pname}': {e}")
for pname in aalop_products:
    if pname not in revenue or pname not in demand or pname not in init_inventory:
        raise ValueError(f"Missing parameter(s) for Aalop product '{pname}'.")
m = gp.Model('Aalop_Inventory_Allocation')
x = m.addVars(aalop_products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in aalop_products)), gp.GRB.MAXIMIZE)
for i in aalop_products:
    m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
for i in aalop_products:
    m.addConstr(x[i] <= init_inventory[i], name=f'inventory_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Optimal Fulfillment Plan for Aalop Products ---')
    for i in aalop_products:
        print(f'Product: {i}')
        print(f'  Units to fulfill (x): {int(round(x[i].X))}')
        print(f'  Revenue per unit: {revenue[i]}')
        print(f'  Demand: {demand[i]}')
        print(f'  Initial Inventory: {init_inventory[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')