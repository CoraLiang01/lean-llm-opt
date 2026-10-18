import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['ShelfID'] = capacity_df['ShelfID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
shelf_ids = capacity_df['ShelfID'].tolist()
capacities = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].astype(int)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
product_ids = products_df['ProductName'].tolist()
values = dict(zip(products_df['ProductName'], products_df['Value']))
weights = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('BigMartSales_Allocation')
x_vars = m.addVars(shelf_ids, product_ids, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x_vars[i, j] for i in shelf_ids for j in product_ids)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x_vars[i, j] for j in product_ids)) <= capacities[i] for i in shelf_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')