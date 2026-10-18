import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
capacity_dict = dict(zip(capacity_df['BookshelfID'], capacity_df['Capacity']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
I = list(capacity_dict.keys())
J = list(weight_dict.keys())
m = gp.Model('BookstoreAllocation')
x = m.addVars(I, J, vtype=gp.GRB.INTEGER)
for i in I:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in J)) <= capacity_dict[i])
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('\n--- Allocation Summary ---')
    for i in I:
        shelf_total = 0
        for j in J:
            qty = int(round(x[i, j].X))
            if qty > 0:
                print(f'Shelf {i}: {qty} units of "{j}"')
                shelf_total += qty
        if shelf_total == 0:
            print(f'Shelf {i}: No books allocated')
else:
    print(f'No optimal solution found. Status: {m.status}')