import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv', sep=',')
capacity_df['ShelfID'] = capacity_df['ShelfID'].astype(int)
shelves = capacity_df['ShelfID'].tolist()
shelf_cap = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(int)
products = products_df['ProductName'].tolist()
prod_value = dict(zip(products_df['ProductName'], products_df['Value']))
prod_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(shelves) != set(capacity_df['ShelfID']):
    raise ValueError('Mismatch in shelf IDs between index set and capacity data.')
if set(products) != set(products_df['ProductName']):
    raise ValueError('Mismatch in product IDs between index set and product data.')

def solve_problem(shelves, products, shelf_cap, prod_value, prod_weight):
    m = gp.Model('BigMartShelfAllocation')
    x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((prod_value[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((prod_weight[j] * x[i, j] for j in products)) <= shelf_cap[i] for i in shelves), name='')
    m.optimize()
    return m
m = solve_problem(shelves, products, shelf_cap, prod_value, prod_weight)