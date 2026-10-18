import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if not {'CabinetID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError("capacity.csv must contain columns 'CabinetID' and 'Capacity'.")
capacity_df['CabinetID'] = capacity_df['CabinetID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
cabinet_ids = capacity_df['CabinetID'].astype(int).tolist()
cabinet_capacities = capacity_df.set_index('CabinetID')['Capacity'].astype(float).to_dict()
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError("products.csv must contain columns 'ProductName', 'Value', and 'Weight'.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
product_values = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
product_weights = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinet_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[prod] * x_vars[cab, prod] for cab in cabinet_ids for prod in product_names)), gp.GRB.MAXIMIZE)
for cab in cabinet_ids:
    m.addConstr(gp.quicksum((product_weights[prod] * x_vars[cab, prod] for prod in product_names)) <= cabinet_capacities[str(cab)], name=f'capacity_cabinet_{cab}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation (cabinet, product, units) ---')
    for cab in cabinet_ids:
        for prod in product_names:
            val = x_vars[cab, prod].X
            if val > 1e-06:
                print(f"Cabinet {cab} - Product '{prod}': {int(round(val))} units")
        total_weight = sum((product_weights[prod] * x_vars[cab, prod].X for prod in product_names))
        print(f'  >> Total weight in Cabinet {cab}: {total_weight:.2f} / {cabinet_capacities[str(cab)]}')
else:
    print(f'No optimal solution found. Status: {m.status}')