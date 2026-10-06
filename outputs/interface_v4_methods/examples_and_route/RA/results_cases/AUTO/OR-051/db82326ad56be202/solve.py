import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv', sep=',')
capacity_df['CabinetID'] = capacity_df['CabinetID'].astype(int)
cabinets = capacity_df['CabinetID'].tolist()
cabinet_cap = dict(zip(capacity_df['CabinetID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products = products_df['ProductName'].tolist()
product_value = dict(zip(products_df['ProductName'], products_df['Value']))
product_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(cabinets) != set(capacity_df['CabinetID']):
    raise ValueError('Mismatch in cabinets between index set and capacity data.')
if set(products) != set(products_df['ProductName']):
    raise ValueError('Mismatch in products between index set and product data.')
m = gp.Model('CoffeeCabinetAllocation')
x = m.addVars(cabinets, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x[i, j] for i in cabinets for j in products)), gp.GRB.MAXIMIZE)
for i in cabinets:
    m.addConstr(gp.quicksum((product_weight[j] * x[i, j] for j in products)) <= cabinet_cap[i], name=f'cap_{i}')
m.optimize()