import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv', sep=',')
display_ids = capacity_df['DisplayID'].astype(int).tolist()
boat_types = products_df['ProductName'].astype(str).tolist()
capacity = dict(zip(capacity_df['DisplayID'].astype(int), capacity_df['Capacity'].astype(int)))
value = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if set(display_ids) != set(capacity.keys()):
    raise ValueError('Mismatch between display_ids and capacity keys.')
if set(boat_types) != set(value.keys()) or set(boat_types) != set(weight.keys()):
    raise ValueError('Mismatch between boat_types and value/weight keys.')
m = gp.Model('BoatDisplayAllocation')
x = m.addVars(display_ids, boat_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in display_ids for j in boat_types)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in boat_types)) <= capacity[i], name=f'cap_{i}')
m.optimize()