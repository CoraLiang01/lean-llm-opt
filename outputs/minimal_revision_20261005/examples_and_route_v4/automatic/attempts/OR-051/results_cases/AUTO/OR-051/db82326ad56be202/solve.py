import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if df_capacity['CabinetID'].isnull().any() or df_capacity['Capacity'].isnull().any():
    raise ValueError('Missing CabinetID or Capacity in capacity.csv')
cabinets = df_capacity['CabinetID'].astype(int).tolist()
capacity_dict = dict(zip(df_capacity['CabinetID'].astype(int), df_capacity['Capacity'].astype(float)))
df_products = pd.read_csv(products_path, sep=',')
if df_products['ProductName'].isnull().any() or df_products['Value'].isnull().any() or df_products['Weight'].isnull().any():
    raise ValueError('Missing ProductName, Value, or Weight in products.csv')
products = df_products['ProductName'].astype(str).tolist()
value_dict = dict(zip(df_products['ProductName'].astype(str), df_products['Value'].astype(float)))
weight_dict = dict(zip(df_products['ProductName'].astype(str), df_products['Weight'].astype(float)))
if set(capacity_dict.keys()) != set(cabinets):
    raise ValueError('Mismatch in cabinets and capacity_dict keys')
if set(value_dict.keys()) != set(products) or set(weight_dict.keys()) != set(products):
    raise ValueError('Mismatch in products and value/weight dict keys')
m = gp.Model('CoffeeCabinetAllocation')
x_keys = [(i, j) for i in cabinets for j in products]
x = m.addVars(x_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in cabinets for j in products)), gp.GRB.MAXIMIZE)
for i in cabinets:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name=f'cap_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in cabinets:
        for j in products:
            var = x[i, j]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')