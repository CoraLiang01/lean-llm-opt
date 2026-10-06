import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv', sep=',')
capacity_df['DisplayID'] = capacity_df['DisplayID'].astype(int)
display_ids = capacity_df['DisplayID'].tolist()
capacity_dict = dict(zip(capacity_df['DisplayID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
boat_types = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(display_ids) == 0:
    raise ValueError('No display areas found in capacity.csv')
if len(boat_types) == 0:
    raise ValueError('No boat types found in products.csv')
if set(capacity_dict.keys()) != set(display_ids):
    raise ValueError('Mismatch in display area IDs between list and capacity_dict')
if set(value_dict.keys()) != set(boat_types) or set(weight_dict.keys()) != set(boat_types):
    raise ValueError('Mismatch in boat type keys between value_dict/weight_dict and boat_types')
m = gp.Model('BoatDisplayAssignment')
x = m.addVars(display_ids, boat_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in display_ids for j in boat_types)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in boat_types)) <= capacity_dict[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Assignment of Boats to Display Areas ---')
    for i in display_ids:
        assigned = []
        for j in boat_types:
            n = int(round(x[i, j].X))
            if n > 0:
                assigned.append((j, n))
        if assigned:
            print(f'Display Area {i} (Capacity {capacity_dict[i]}):')
            for j, n in assigned:
                print(f'  {j}: {n} units (Value per unit: {value_dict[j]}, Weight per unit: {weight_dict[j]})')
        else:
            print(f'Display Area {i} (Capacity {capacity_dict[i]}): No boats assigned.')
else:
    print(f'No optimal solution found. Status: {m.status}')