import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['ShelfID'] = capacity_df['ShelfID'].astype(str)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
shelves = list(capacity_df['ShelfID'])
capacity = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = products_df['Value'].astype(float)
products_df['Weight'] = products_df['Weight'].astype(float)
products = list(products_df['ProductName'])
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(shelves) == 0 or len(products) == 0:
    raise ValueError('No shelves or products found in the input data.')
for s in shelves:
    if s not in capacity:
        raise ValueError(f'Missing capacity for shelf {s}.')
for p in products:
    if p not in value or p not in weight:
        raise ValueError(f'Missing value or weight for product {p}.')
m = gp.Model('RetailShelfAllocation')
x_vars = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
for s in shelves:
    m.addConstr(gp.quicksum((weight[p] * x_vars[s, p] for p in products)) <= capacity[s], name=f'cap_{s}')
first_product = products[0]
m.addConstr(gp.quicksum((x_vars[s, first_product] for s in shelves)) >= 5, name='min_smartphone')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')