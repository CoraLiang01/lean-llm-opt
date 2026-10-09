import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
if not {'DisplayID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError('capacity.csv must contain columns: DisplayID, Capacity')
capacity_df['DisplayID'] = capacity_df['DisplayID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
capacity_df['DisplayID_int'] = capacity_df['DisplayID'].astype(int)
capacity_df['Capacity_int'] = capacity_df['Capacity'].astype(int)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
products_df['Value_int'] = products_df['Value'].astype(int)
products_df['Weight_int'] = products_df['Weight'].astype(int)
display_ids = list(capacity_df['DisplayID_int'])
boat_types = list(products_df['ProductName'])
capacity_dict = dict(zip(capacity_df['DisplayID_int'], capacity_df['Capacity_int']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value_int']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight_int']))
if len(capacity_dict) != len(display_ids):
    raise ValueError('Mismatch in display area count and capacity_dict keys.')
if len(value_dict) != len(boat_types) or len(weight_dict) != len(boat_types):
    raise ValueError('Mismatch in boat type count and value/weight dict keys.')
m = gp.Model('BoatDisplayAllocation')
x_vars = m.addVars(display_ids, boat_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in display_ids for j in boat_types)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in boat_types)) <= capacity_dict[i], name=f'cap_{i}')
m.optimize()