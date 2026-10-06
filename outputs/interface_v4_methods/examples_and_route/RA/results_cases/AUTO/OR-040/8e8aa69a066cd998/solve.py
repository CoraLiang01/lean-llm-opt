import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
capacity_df = pd.read_csv(capacity_path, sep=',')
areas = products_df['ProductName'].astype(str).tolist()
value_dict = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight_dict = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
total_capacity = int(capacity_df['Capacity'].iloc[0])
for area in areas:
    if area not in value_dict or area not in weight_dict:
        raise ValueError(f"Missing Value or Weight for area '{area}' in products.csv.")

def solve_nyc_development_knapsack(areas, value_dict, weight_dict, total_capacity):
    m = gp.Model('NYC_Development_Knapsack')
    x = m.addVars(areas, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[area] * x[area] for area in areas)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[area] * x[area] for area in areas)) <= total_capacity, name='capacity')
    m.optimize()
    return m
m = solve_nyc_development_knapsack(areas, value_dict, weight_dict, total_capacity)