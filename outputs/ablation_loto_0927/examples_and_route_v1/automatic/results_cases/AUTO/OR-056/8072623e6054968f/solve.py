import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
display_ids = capacity_df['DisplayID'].astype(int).tolist()
boat_types = products_df['ProductName'].astype(str).tolist()
capacity_dict = dict(zip(capacity_df['DisplayID'].astype(int), capacity_df['Capacity'].astype(int)))
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if set(display_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between display_ids and capacity_dict keys.')
if set(boat_types) != set(value_dict.keys()) or set(boat_types) != set(weight_dict.keys()):
    raise ValueError('Mismatch between boat_types and value/weight dict keys.')
m = gp.Model('BoatDisplayAssignment')
x = m.addVars(display_ids, boat_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in display_ids for j in boat_types)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in boat_types)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Assignment of Boats to Display Areas ---')
    for i in display_ids:
        assigned = []
        for j in boat_types:
            qty = int(round(x[i, j].X))
            if qty > 0:
                assigned.append((j, qty))
        if assigned:
            print(f'Display Area {i} (Capacity {capacity_dict[i]}):')
            for (j, qty) in assigned:
                print(f'  {j}: {qty} units (Value per unit: {value_dict[j]}, Weight per unit: {weight_dict[j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')