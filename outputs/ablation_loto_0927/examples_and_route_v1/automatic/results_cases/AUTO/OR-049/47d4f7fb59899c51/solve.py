import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
shelves = capacity_df['ShelfID'].astype(int).tolist()
shelf_capacity = dict(zip(capacity_df['ShelfID'].astype(int), capacity_df['Capacity'].astype(float)))
products = products_df['ProductName'].astype(str).tolist()
product_value = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
product_weight = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(float)))
if len(shelves) != len(shelf_capacity):
    raise ValueError('Mismatch in shelf capacity data.')
if len(products) != len(product_value) or len(products) != len(product_weight):
    raise ValueError('Mismatch in product value/weight data.')
m = gp.Model('RetailShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((product_weight[j] * x[i, j] for j in products)) <= shelf_capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Shelf Allocations (x[i,j]) ---')
    for i in shelves:
        shelf_total_value = 0
        shelf_total_weight = 0
        print(f'Shelf {i}:')
        for j in products:
            units = x[i, j].X
            if units > 1e-06:
                value = product_value[j] * units
                weight = product_weight[j] * units
                shelf_total_value += value
                shelf_total_weight += weight
                print(f'  {j}: {int(round(units))} units (Value: {value:.2f}, Weight: {weight:.2f})')
        print(f'  >> Total value: {shelf_total_value:.2f}, Total weight: {shelf_total_weight:.2f} / Capacity: {shelf_capacity[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')