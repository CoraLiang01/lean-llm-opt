import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
product_ids = products_df['ProductName'].tolist()
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'Capacity' not in capacity_df.columns:
    raise KeyError('capacity.csv must contain column: Capacity')
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
capacity = int(capacity_df['Capacity'].iloc[0])

def solve_problem(product_ids, value, weight, capacity):
    m = gp.Model('PharmacyDrugOrder')
    x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_ids)) <= capacity, name='capacity')
    m.optimize()
    return m
m = solve_problem(product_ids, value, weight, capacity)