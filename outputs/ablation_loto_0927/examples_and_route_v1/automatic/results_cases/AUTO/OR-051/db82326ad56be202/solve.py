import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
cabinets = capacity_df['CabinetID'].astype(int).tolist()
products = products_df['ProductName'].astype(str).tolist()
capacity_dict = dict(zip(capacity_df['CabinetID'].astype(int), capacity_df['Capacity'].astype(float)))
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(float)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(float)))
if set(cabinets) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between cabinets index set and capacity_dict keys.')
if set(products) != set(value_dict.keys()) or set(products) != set(weight_dict.keys()):
    raise ValueError('Mismatch between products index set and value/weight dict keys.')
m = gp.Model('CoffeeCabinetAllocation')
x = m.addVars(cabinets, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in cabinets for j in products)), gp.GRB.MAXIMIZE)
for i in cabinets:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Allocation Plan (units of each product in each cabinet) ---')
    for i in cabinets:
        print(f'Cabinet {i} (Capacity: {capacity_dict[i]:.1f}):')
        total_weight = 0.0
        total_value = 0.0
        for j in products:
            units = x[i, j].X
            if units > 1e-06:
                w = weight_dict[j] * units
                v = value_dict[j] * units
                print(f'  {j}: {int(round(units))} units (Weight: {w:.1f}, Value: {v:.1f})')
                total_weight += w
                total_value += v
        print(f'  >> Total weight used: {total_weight:.1f} / {capacity_dict[i]:.1f}')
        print(f'  >> Total value in cabinet: {total_value:.1f}')
else:
    print(f'No optimal solution found. Status: {m.status}')