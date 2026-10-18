import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['SectionID'] = capacity_df['SectionID'].astype(str).str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
sections = list(capacity_df['SectionID'])
capacity = dict(zip(capacity_df['SectionID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(float)
products_df['Weight'] = products_df['Weight'].astype(float)
products = list(products_df['ProductName'])
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('Supermarket_Section_Allocation')
x_vars = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in sections for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in products)) <= capacity[i] for i in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')