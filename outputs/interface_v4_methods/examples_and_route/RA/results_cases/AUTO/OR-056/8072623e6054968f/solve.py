import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv', sep=',')
if not {'DisplayID', 'Capacity'}.issubset(capacity_df.columns):
    raise ValueError('capacity.csv must contain columns: DisplayID, Capacity')
capacity_df['DisplayID'] = capacity_df['DisplayID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
display_ids = list(capacity_df['DisplayID'])
display_capacities = dict(zip(capacity_df['DisplayID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv', sep=',')
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
product_names = list(products_df['ProductName'])
product_values = dict(zip(products_df['ProductName'], products_df['Value']))
product_weights = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(display_ids) != len(set(display_ids)):
    raise ValueError('DisplayID values in capacity.csv must be unique')
if len(product_names) != len(set(product_names)):
    raise ValueError('ProductName values in products.csv must be unique')

def solve_boat_display_assignment(display_ids, display_capacities, product_names, product_values, product_weights):
    m = gp.Model('BoatDisplayAssignment')
    x = m.addVars(display_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((product_values[j] * x[i, j] for i in display_ids for j in product_names)), gp.GRB.MAXIMIZE)
    for i in display_ids:
        m.addConstr(gp.quicksum((product_weights[j] * x[i, j] for j in product_names)) <= display_capacities[i], name=f'cap_{i}')
    m.optimize()
    return m
m = solve_boat_display_assignment(display_ids, display_capacities, product_names, product_values, product_weights)