import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
    products_df = pd.read_csv(products_path, sep=',')
    capacity_df = pd.read_csv(capacity_path, sep=',')
    areas = products_df['ProductName'].astype(str).tolist()
    value = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
    weight = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
        raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
    capacity = float(capacity_df['Capacity'].iloc[0])
    for area in areas:
        if area not in value or area not in weight:
            raise ValueError(f"Missing value or weight for area '{area}'.")
    m = gp.Model('NYC_RealEstate_Development')
    x = m.addVars(areas, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((value[area] * x[area] for area in areas)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[area] * x[area] for area in areas)) <= capacity, name='capacity')
    m.optimize()
    return m
m = solve_problem()