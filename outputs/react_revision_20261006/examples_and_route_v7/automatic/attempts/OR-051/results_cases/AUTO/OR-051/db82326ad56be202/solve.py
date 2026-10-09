import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if 'CabinetID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain columns 'CabinetID' and 'Capacity'")
cabinet_ids = capacity_df['CabinetID'].astype(str).tolist()
cabinet_id_set = set(cabinet_ids)
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    cab = str(row['CabinetID'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f'Non-integer Capacity for CabinetID {cab}')
    capacity_dict[cab] = cap
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain columns 'ProductName', 'Value', and 'Weight'")
product_names = products_df['ProductName'].astype(str).tolist()
product_name_set = set(product_names)
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = str(row['ProductName'])
    try:
        val = int(row['Value'])
    except Exception:
        raise ValueError(f'Non-integer Value for ProductName {pname}')
    try:
        wgt = float(row['Weight'])
    except Exception:
        raise ValueError(f'Non-numeric Weight for ProductName {pname}')
    value_dict[pname] = val
    weight_dict[pname] = wgt
if set(capacity_dict.keys()) != set(cabinet_ids):
    raise ValueError('Mismatch in CabinetID keys in capacity_dict and cabinet_ids')
if set(value_dict.keys()) != set(product_names) or set(weight_dict.keys()) != set(product_names):
    raise ValueError('Mismatch in ProductName keys in value_dict/weight_dict and product_names')
decision_keys = [(cab, prod) for cab in cabinet_ids for prod in product_names]
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[prod] * x_vars[cab, prod] for cab in cabinet_ids for prod in product_names)), gp.GRB.MAXIMIZE)
for cab in cabinet_ids:
    m.addConstr(gp.quicksum((weight_dict[prod] * x_vars[cab, prod] for prod in product_names)) <= capacity_dict[cab], name=f'cap_{cab}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for cab in cabinet_ids:
        for prod in product_names:
            var = x_vars[cab, prod]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')