import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', sep=',')
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise ValueError("products.csv must have columns: 'ProductName', 'Value', 'Weight'.")
areas = products_df['ProductName'].astype(str).tolist()
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(float)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(float)))
if set(areas) != set(value_dict.keys()) or set(areas) != set(weight_dict.keys()):
    raise ValueError('Mismatch in area identifiers between products.csv columns.')

def solve_problem(areas, value_dict, weight_dict, capacity):
    m = gp.Model('NYC_RealEstate_Development')
    m.Params.MIPGap = 0.0001
    x = m.addVars(areas, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in areas)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in areas)) <= capacity, name='cap')
    m.optimize()
    return m
m = solve_problem(areas, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')