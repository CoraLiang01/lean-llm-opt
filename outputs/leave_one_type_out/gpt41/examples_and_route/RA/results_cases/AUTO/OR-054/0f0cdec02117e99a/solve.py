import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
shelves = capacity_df['ShelfID'].astype(int).tolist()
products = products_df['ProductName'].astype(int).tolist()
capacity = dict(zip(capacity_df['ShelfID'].astype(int), capacity_df['Capacity'].astype(int)))
value = dict(zip(products_df['ProductName'].astype(int), products_df['Value'].astype(int)))
weight = dict(zip(products_df['ProductName'].astype(int), products_df['Weight'].astype(int)))
if set(shelves) != set(capacity.keys()):
    raise ValueError('Mismatch between shelves index set and capacity keys.')
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch between products index set and value/weight keys.')
m = gp.Model('BigMartShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Shelf Allocation ---')
    for i in shelves:
        shelf_total_value = 0
        shelf_total_weight = 0
        allocation = []
        for j in products:
            units = x[i, j].X
            if units >= 1e-06:
                allocation.append((j, int(round(units)), value[j], weight[j]))
                shelf_total_value += value[j] * units
                shelf_total_weight += weight[j] * units
        print(f'Shelf {i}:')
        print(f'  Total Value: {shelf_total_value:.2f}')
        print(f'  Total Weight: {shelf_total_weight:.2f} / Capacity {capacity[i]}')
        if allocation:
            print(f'  Product Allocations:')
            for j, units, v, w in allocation:
                print(f'    Product {j}: {units} units (Value {v}, Weight {w})')
        else:
            print('  No products allocated.')
else:
    print(f'No optimal solution found. Status: {m.status}')