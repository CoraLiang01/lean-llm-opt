import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df['DisplayID'].isnull().any() or capacity_df['Capacity'].isnull().any():
    raise ValueError('Missing DisplayID or Capacity in capacity.csv')
display_ids = capacity_df['DisplayID'].astype(int).tolist()
display_capacities = dict(zip(capacity_df['DisplayID'].astype(int), capacity_df['Capacity'].astype(int)))
products_df = pd.read_csv(products_path, sep=',')
if products_df['ProductName'].isnull().any() or products_df['Value'].isnull().any() or products_df['Weight'].isnull().any():
    raise ValueError('Missing ProductName, Value, or Weight in products.csv')
product_names = products_df['ProductName'].astype(str).tolist()
product_values = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
product_weights = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if set(display_ids) != set(capacity_df['DisplayID']):
    raise ValueError('Mismatch in display area identifiers.')
if set(product_names) != set(products_df['ProductName']):
    raise ValueError('Mismatch in product identifiers.')
decision_keys = [(i, j) for i in display_ids for j in product_names]
m = gp.Model('BoatDisplayAllocation')
x = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x[i, j] for (i, j) in decision_keys)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x[i, j] for j in product_names)) <= display_capacities[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in decision_keys:
        var = x[i, j]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')