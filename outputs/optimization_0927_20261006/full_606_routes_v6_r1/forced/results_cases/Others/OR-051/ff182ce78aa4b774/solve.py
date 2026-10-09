import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if not {'CabinetID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError("capacity.csv must contain columns 'CabinetID' and 'Capacity'.")
capacity_df['CabinetID'] = capacity_df['CabinetID'].str.strip().astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip().astype(float)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError("products.csv must contain columns 'ProductName', 'Value', and 'Weight'.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip().astype(float)
products_df['Weight'] = products_df['Weight'].str.strip().astype(float)
cabinets = capacity_df['CabinetID'].tolist()
products = products_df['ProductName'].tolist()
capacity_dict = dict(zip(capacity_df['CabinetID'], capacity_df['Capacity']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(capacity_dict) != len(cabinets):
    raise ValueError('Mismatch in number of cabinets and capacity entries.')
if len(value_dict) != len(products) or len(weight_dict) != len(products):
    raise ValueError('Mismatch in number of products and value/weight entries.')
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinets, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in cabinets for j in products)), gp.GRB.MAXIMIZE)
for i in cabinets:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in products)) <= capacity_dict[i], name=f'CabinetCapacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation Plan (units per cabinet per product) ---')
    for i in cabinets:
        print(f'Cabinet {i} (Capacity {capacity_dict[i]:.0f}):')
        total_weight = 0.0
        total_value = 0.0
        for j in products:
            units = x_vars[i, j].X
            if units > 1e-06:
                weight = weight_dict[j] * units
                value = value_dict[j] * units
                print(f'  {j}: {units:.0f} units (Weight {weight:.1f}, Value {value:.0f})')
                total_weight += weight
                total_value += value
        print(f'  >> Total weight used: {total_weight:.1f} / {capacity_dict[i]:.0f}')
        print(f'  >> Total value in cabinet: {total_value:.0f}')
else:
    print(f'No optimal solution found. Status: {m.status}')