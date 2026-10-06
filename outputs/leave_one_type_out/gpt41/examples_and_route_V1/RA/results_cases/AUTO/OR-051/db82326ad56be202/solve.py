import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_capacity['CabinetID'] = df_capacity['CabinetID'].astype(int)
cabinets = df_capacity['CabinetID'].tolist()
cabinet_cap = dict(zip(df_capacity['CabinetID'], df_capacity['Capacity']))
df_products = pd.read_csv(products_path, sep=',')
df_products['ProductName'] = df_products['ProductName'].astype(str).str.strip()
products = df_products['ProductName'].tolist()
product_value = dict(zip(df_products['ProductName'], df_products['Value']))
product_weight = dict(zip(df_products['ProductName'], df_products['Weight']))
if len(cabinets) != len(set(cabinets)):
    raise ValueError('Duplicate CabinetID found in capacity.csv')
if len(products) != len(set(products)):
    raise ValueError('Duplicate ProductName found in products.csv')
if set(cabinet_cap.keys()) != set(cabinets):
    raise ValueError('Mismatch in CabinetID keys')
if set(product_value.keys()) != set(products) or set(product_weight.keys()) != set(products):
    raise ValueError('Mismatch in ProductName keys for value/weight')
m = gp.Model('CoffeeCabinetAllocation')
x = m.addVars(cabinets, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x[i, j] for i in cabinets for j in products)), gp.GRB.MAXIMIZE)
for i in cabinets:
    m.addConstr(gp.quicksum((product_weight[j] * x[i, j] for j in products)) <= cabinet_cap[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Allocation Plan (units of each product in each cabinet) ---')
    for i in cabinets:
        print(f'Cabinet {i} (Capacity: {cabinet_cap[i]}):')
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
        print(f'  >> Total weight used: {total_weight:.1f} / {cabinet_cap[i]}, Total value: {total_value:.1f}')
else:
    print(f'No optimal solution found. Status: {m.status}')