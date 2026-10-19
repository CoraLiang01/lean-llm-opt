import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
cabinet_ids = capacity_df['CabinetID'].tolist()
capacities = dict(zip(cabinet_ids, capacity_df['Capacity']))
product_names = products_df['ProductName'].tolist()
values = dict(zip(product_names, products_df['Value']))
weights = dict(zip(product_names, products_df['Weight']))
product_index = {name: idx for idx, name in enumerate(product_names)}
value_list = [values[name] for name in product_names]
weight_list = [weights[name] for name in product_names]
num_products = len(product_names)
m = gp.Model('CoffeeCabinetAllocation')
x = m.addVars(cabinet_ids, range(num_products), vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value_list[j] * x[i, j] for i in cabinet_ids for j in range(num_products))), gp.GRB.MAXIMIZE)
for i in cabinet_ids:
    m.addConstr(gp.quicksum((weight_list[j] * x[i, j] for j in range(num_products))) <= capacities[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    for i in cabinet_ids:
        allocation = [(product_names[j], x[i, j].X) for j in range(num_products) if x[i, j].X > 0]
        if allocation:
            print(f'Cabinet {i} (Capacity {capacities[i]}):')
            for pname, qty in allocation:
                print(f'  {pname}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')