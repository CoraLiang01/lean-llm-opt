import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', sep=',')
capacity_df['StorageID'] = capacity_df['StorageID'].astype(int)
storage_ids = capacity_df['StorageID'].tolist()
capacity_dict = dict(zip(capacity_df['StorageID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
I = storage_ids
J = product_names
if len(I) != len(capacity_dict):
    raise ValueError('Mismatch in storage area count between index set and capacity dictionary.')
if len(J) != len(value_dict) or len(J) != len(weight_dict):
    raise ValueError('Mismatch in product count between index set and value/weight dictionaries.')
m = gp.Model('Amazon_AC_Storage_Allocation')
x = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in J)) <= capacity_dict[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Allocation Plan ---')
    for i in I:
        assigned = []
        for j in J:
            qty = x[i, j].X
            if qty >= 1e-06:
                assigned.append((j, int(round(qty))))
        if assigned:
            print(f'Storage Area {i}:')
            for j, qty in assigned:
                print(f'  {j}: {qty} units')
else:
    print(f'No optimal solution found. Status: {m.status}')