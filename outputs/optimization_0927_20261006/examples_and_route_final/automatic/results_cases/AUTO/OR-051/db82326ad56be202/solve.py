import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv', dtype=str, keep_default_na=False)
cabinet_ids = capacity_df['CabinetID'].astype(int).tolist()
cabinet_capacity = dict(zip(capacity_df['CabinetID'].astype(int), capacity_df['Capacity'].astype(float)))
product_names = products_df['ProductName'].astype(str).tolist()
product_value = dict(zip(products_df['ProductName'], products_df['Value'].astype(float)))
product_weight = dict(zip(products_df['ProductName'], products_df['Weight'].astype(float)))
if set(cabinet_capacity.keys()) != set(cabinet_ids):
    raise ValueError('Mismatch in cabinet IDs between index set and capacity mapping.')
if set(product_value.keys()) != set(product_names) or set(product_weight.keys()) != set(product_names):
    raise ValueError('Mismatch in product names between index set and value/weight mapping.')
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinet_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x_vars[i, j] for i in cabinet_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in cabinet_ids:
    m.addConstr(gp.quicksum((product_weight[j] * x_vars[i, j] for j in product_names)) <= cabinet_capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation Plan (units per cabinet per product) ---')
    for i in cabinet_ids:
        print(f'Cabinet {i} (Capacity: {cabinet_capacity[i]:.1f}):')
        total_weight = 0.0
        total_value = 0.0
        for j in product_names:
            units = x_vars[i, j].X
            if units > 1e-06:
                weight = product_weight[j] * units
                value = product_value[j] * units
                print(f'  {j}: {int(round(units))} units (Weight: {weight:.1f}, Value: {value:.1f})')
                total_weight += weight
                total_value += value
        print(f'  >> Total weight used: {total_weight:.1f} / {cabinet_capacity[i]:.1f}')
        print(f'  >> Total value in cabinet: {total_value:.1f}')
else:
    print(f'No optimal solution found. Status: {m.status}')