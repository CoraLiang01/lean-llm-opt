import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
capacity_df['DisplayID'] = capacity_df['DisplayID'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
display_ids = capacity_df['DisplayID'].tolist()
product_names = products_df['ProductName'].tolist()
capacity = dict(zip(capacity_df['DisplayID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(display_ids) != set(capacity.keys()):
    raise ValueError('Mismatch between display_ids and capacity keys.')
if set(product_names) != set(value.keys()) or set(product_names) != set(weight.keys()):
    raise ValueError('Mismatch between product_names and value/weight keys.')
m = gp.Model('BoatDisplayAssignment')
x = m.addVars(display_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in display_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in product_names)) <= capacity[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Assignment of Boats to Display Areas ---')
    for i in display_ids:
        assigned = []
        for j in product_names:
            qty = x[i, j].X
            if qty >= 1e-06:
                assigned.append((j, int(round(qty))))
        if assigned:
            print(f'Display Area {i}:')
            for j, qty in assigned:
                print(f'  {j}: {qty}')
else:
    print(f'No optimal solution found. Status: {m.status}')