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
cabinet_ids = capacity_df['CabinetID'].tolist()
product_names = products_df['ProductName'].tolist()
cabinet_capacity = dict(zip(capacity_df['CabinetID'], capacity_df['Capacity']))
product_value = dict(zip(products_df['ProductName'], products_df['Value']))
product_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinet_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x_vars[i, j] for i in cabinet_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in cabinet_ids:
    m.addConstr(gp.quicksum((product_weight[j] * x_vars[i, j] for j in product_names)) <= cabinet_capacity[i], name=f'cabinet_capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation per Cabinet ---')
    for i in cabinet_ids:
        print(f'Cabinet {i}:')
        total_weight = 0.0
        total_value = 0.0
        for j in product_names:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f'  {j}: {qty:.0f} units (Value: {product_value[j]:.0f}, Weight: {product_weight[j]:.2f})')
                total_weight += qty * product_weight[j]
                total_value += qty * product_value[j]
        print(f'  >> Total weight: {total_weight:.2f} / {cabinet_capacity[i]:.2f}')
        print(f'  >> Total value: {total_value:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')