import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_capacity['ShelfID'] = df_capacity['ShelfID'].astype(int)
df_capacity['Capacity'] = df_capacity['Capacity'].astype(int)
shelves = df_capacity['ShelfID'].tolist()
capacity = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
df_products = pd.read_csv(products_path, sep=',')
df_products['ProductName'] = df_products['ProductName'].astype(int)
df_products['Value'] = df_products['Value'].astype(int)
df_products['Weight'] = df_products['Weight'].astype(int)
products = df_products['ProductName'].tolist()
value = dict(zip(df_products['ProductName'], df_products['Value']))
weight = dict(zip(df_products['ProductName'], df_products['Weight']))
if len(shelves) == 0 or len(products) == 0:
    raise ValueError('No shelves or products found in the input data.')
if set(capacity.keys()) != set(shelves):
    raise ValueError('Mismatch in shelf IDs between capacity data and shelf list.')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Mismatch in product IDs between value/weight data and product list.')
m = gp.Model('BigMart_Shelf_Allocation')
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
            if units > 1e-06:
                allocation.append((j, int(round(units)), value[j], weight[j]))
                shelf_total_value += value[j] * units
                shelf_total_weight += weight[j] * units
        print(f'Shelf {i}:')
        print(f'  Total Value: {shelf_total_value:.0f}, Total Weight: {shelf_total_weight:.0f} / {capacity[i]}')
        if allocation:
            print('  Product Allocations:')
            for (j, units, v, w) in allocation:
                print(f'    Product {j}: {units} units (Value {v}, Weight {w})')
        else:
            print('  No products allocated.')
else:
    print(f'No optimal solution found. Status: {m.status}')