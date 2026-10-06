import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row with a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv must contain columns: {required_cols}')
products_df = products_df.copy()
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
products = list(products_df['ProductName'])
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))

def solve_problem(products, value, weight, capacity):
    m = gp.Model('PharmacyRestock')
    x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in products)) <= capacity, name='capacity')
    m.optimize()
    return m
m = solve_problem(products, value, weight, capacity)