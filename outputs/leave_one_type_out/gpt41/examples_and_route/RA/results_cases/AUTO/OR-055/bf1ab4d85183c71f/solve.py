import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv', sep=',')
capacity_df['DisplayID'] = capacity_df['DisplayID'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(str)
display_ids = capacity_df['DisplayID'].tolist()
boat_types = products_df['ProductName'].tolist()
capacity = dict(zip(capacity_df['DisplayID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(display_ids) != set(capacity.keys()):
    raise ValueError('Mismatch between display area indices and capacity keys.')
if set(boat_types) != set(value.keys()) or set(boat_types) != set(weight.keys()):
    raise ValueError('Mismatch between boat type indices and value/weight keys.')
m = gp.Model('BoatDisplayAllocation')
x = m.addVars(display_ids, boat_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in display_ids for j in boat_types)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in boat_types)) <= capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Allocation Plan (Display Area x Boat Type) ---')
    for i in display_ids:
        area_total_value = 0
        area_total_weight = 0
        allocation = []
        for j in boat_types:
            qty = int(round(x[i, j].X))
            if qty > 0:
                allocation.append((j, qty, value[j], weight[j]))
                area_total_value += value[j] * qty
                area_total_weight += weight[j] * qty
        if allocation:
            print(f'\nDisplay Area {i} (Capacity: {capacity[i]})')
            print(f'  Total Value: {area_total_value}')
            print(f'  Total Weight Used: {area_total_weight}')
            for j, qty, v, w in allocation:
                print(f'    {j}: {qty} units (Value: {v}, Weight: {w})')
else:
    print(f'No optimal solution found. Status: {m.status}')