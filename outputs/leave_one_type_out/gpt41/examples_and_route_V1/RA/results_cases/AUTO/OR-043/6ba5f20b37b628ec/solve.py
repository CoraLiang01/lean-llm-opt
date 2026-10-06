import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv', sep=',')
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv', sep=',')
product_ids = products_df['ProductName'].astype(str).tolist()
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("Capacity file must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
m = gp.Model('PharmacyDrugOrder')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_ids)) <= capacity, name='stock_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/benefit: {m.objVal:.0f}')
    print('--- Order Plan ---')
    for i in product_ids:
        qty = x[i].X
        if qty > 0.5:
            print(f'{i}: {int(round(qty))} units (Value/unit: {value[i]}, Weight/unit: {weight[i]})')
    total_weight = sum((weight[i] * x[i].X for i in product_ids))
    print(f'Total stock used: {int(round(total_weight))} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')