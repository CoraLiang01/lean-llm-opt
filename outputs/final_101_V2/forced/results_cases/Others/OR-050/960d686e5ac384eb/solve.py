import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv', sep=',')
shelves = capacity_df['ShelfID'].astype(int).tolist()
capacity = dict(zip(capacity_df['ShelfID'].astype(int), capacity_df['Capacity'].astype(float)))
products = products_df['ProductName'].astype(str).tolist()
value = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(float)))
weight = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(float)))
if len(products) == 0:
    raise ValueError('No products found in products.csv')
first_product = products[0]
m = gp.Model('RetailDisplayAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i], name=f'cap_{i}')
m.addConstr(gp.quicksum((x[i, first_product] for i in shelves)) >= 5, name='min_first_product')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation Plan (x[i,j]) ---')
    for i in shelves:
        for j in products:
            v = x[i, j].X
            if v > 1e-06:
                print(f"Shelf {i}, Product '{j}': {int(round(v))} units")
    print(f"\nTotal units of first product ('{first_product}') placed: {int(round(sum((x[i, first_product].X for i in shelves))))}")
else:
    print(f'No optimal solution found. Status: {m.status}')