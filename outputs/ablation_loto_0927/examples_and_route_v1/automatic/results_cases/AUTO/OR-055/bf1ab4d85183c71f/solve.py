import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv', sep=',')
capacity_df['DisplayID'] = capacity_df['DisplayID'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(str)
display_areas = list(capacity_df['DisplayID'])
boat_types = list(products_df['ProductName'])
capacity = dict(zip(capacity_df['DisplayID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(display_areas) != len(capacity):
    raise ValueError('Mismatch in number of display areas and capacities.')
if len(boat_types) != len(value) or len(boat_types) != len(weight):
    raise ValueError('Mismatch in number of boat types and their value/weight data.')
m = gp.Model('BoatDisplayAllocation')
x = m.addVars(display_areas, boat_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in display_areas for j in boat_types)), gp.GRB.MAXIMIZE)
for i in display_areas:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in boat_types)) <= capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Allocation Plan (Display Area, Boat Type, Units) ---')
    for i in display_areas:
        for j in boat_types:
            units = x[i, j].X
            if units > 0.5:
                print(f"Display Area {i} | Boat Type '{j}': {int(round(units))} units")
else:
    print(f'No optimal solution found. Status: {m.status}')