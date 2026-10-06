import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
cabinets = capacity_df['CabinetID'].astype(int).tolist()
cabinet_capacity = dict(zip(capacity_df['CabinetID'].astype(int), capacity_df['Capacity']))
products = products_df['ProductName'].astype(str).tolist()
product_value = dict(zip(products_df['ProductName'].astype(str), products_df['Value']))
product_weight = dict(zip(products_df['ProductName'].astype(str), products_df['Weight']))
if set(cabinets) != set(capacity_df['CabinetID'].astype(int)):
    raise ValueError('Mismatch in cabinet IDs between index set and capacity data.')
if set(products) != set(products_df['ProductName'].astype(str)):
    raise ValueError('Mismatch in product names between index set and product data.')
m = gp.Model('CoffeeCabinetAllocation')
x = m.addVars(cabinets, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x[i, j] for i in cabinets for j in products)), gp.GRB.MAXIMIZE)
for i in cabinets:
    m.addConstr(gp.quicksum((product_weight[j] * x[i, j] for j in products)) <= cabinet_capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Allocation Plan (units of each product in each cabinet) ---')
    for i in cabinets:
        print(f'Cabinet {i} (Capacity: {cabinet_capacity[i]}):')
        total_weight = 0.0
        total_value = 0.0
        for j in products:
            units = x[i, j].X
            if units > 1e-06:
                weight = product_weight[j] * units
                value = product_value[j] * units
                print(f'  {j}: {int(round(units))} units (Weight: {weight:.1f}, Value: {value:.1f})')
                total_weight += weight
                total_value += value
        print(f'  >> Total weight used: {total_weight:.1f} / {cabinet_capacity[i]}')
        print(f'  >> Total value in cabinet: {total_value:.1f}')
else:
    print(f'No optimal solution found. Status: {m.status}')