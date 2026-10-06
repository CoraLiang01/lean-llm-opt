import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv', sep=',')
shelf_ids = capacity_df['ShelfID'].astype(int).tolist()
product_ids = products_df['ProductName'].astype(int).tolist()
capacity_dict = dict(zip(capacity_df['ShelfID'].astype(int), capacity_df['Capacity'].astype(int)))
value_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Weight'].astype(int)))
if set(shelf_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between shelf_ids and capacity_dict keys.')
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch between product_ids and value/weight dict keys.')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(shelf_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelf_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in product_ids)) <= capacity_dict[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Shelf Allocation ---')
    for i in shelf_ids:
        shelf_total_value = 0
        shelf_total_weight = 0
        allocation = []
        for j in product_ids:
            units = int(round(x[i, j].X))
            if units > 0:
                allocation.append((j, units, value_dict[j], weight_dict[j]))
                shelf_total_value += value_dict[j] * units
                shelf_total_weight += weight_dict[j] * units
        print(f'Shelf {i}:')
        print(f'  Total Value: {shelf_total_value}')
        print(f'  Total Weight: {shelf_total_weight} / {capacity_dict[i]}')
        if allocation:
            print(f'  Products placed:')
            for j, units, val, wt in allocation:
                print(f'    Product {j}: {units} units (Value/unit={val}, Weight/unit={wt})')
        else:
            print('  No products placed.')
else:
    print(f'No optimal solution found. Status: {m.status}')