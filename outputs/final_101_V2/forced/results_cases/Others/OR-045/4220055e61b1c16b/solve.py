import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row with a 'Capacity' column.")
total_capacity = int(capacity_df['Capacity'].iloc[0])
required_cols = {'ProductName', 'Weight', 'Value'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv must contain columns: {required_cols}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Weight'] = products_df['Weight'].astype(int)
products_df['Value'] = products_df['Value'].astype(int)
products = list(products_df['ProductName'])
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
m = gp.Model('SupermarketRestock')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in products)) <= total_capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Order Quantities ---')
    for i in products:
        print(f'{i}: {int(round(x[i].X))} units')
else:
    print(f'No optimal solution found. Status: {m.status}')