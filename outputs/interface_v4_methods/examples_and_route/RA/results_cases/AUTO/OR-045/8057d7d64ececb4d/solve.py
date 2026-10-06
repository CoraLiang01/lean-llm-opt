import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    cap_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', sep=',')
    if 'Capacity' not in cap_df.columns or cap_df.shape[0] != 1:
        raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
    capacity = int(cap_df.loc[0, 'Capacity'])
    prod_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', sep=',')
    for col in ['ProductName', 'Weight', 'Value']:
        if col not in prod_df.columns:
            raise ValueError(f'products.csv missing required column: {col}')
    prod_df['ProductName'] = prod_df['ProductName'].astype(str)
    prod_df['Weight'] = pd.to_numeric(prod_df['Weight'], errors='raise')
    prod_df['Value'] = pd.to_numeric(prod_df['Value'], errors='raise')
    products = prod_df['ProductName'].tolist()
    weight = dict(zip(prod_df['ProductName'], prod_df['Weight']))
    value = dict(zip(prod_df['ProductName'], prod_df['Value']))
    m = gp.Model('SupermarketRestock')
    x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in products)) <= capacity, name='capacity')
    m.optimize()
    return m
m = solve_problem()