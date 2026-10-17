import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['CabinetID'] = capacity_df['CabinetID'].astype(str).str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
cabinets = list(capacity_df['CabinetID'])
capacities = dict(zip(capacity_df['CabinetID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv', dtype=str, keep_default_na=False, skipinitialspace=True)
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(float)
products_df['Weight'] = products_df['Weight'].astype(float)
products = list(products_df['ProductName'])
values = dict(zip(products_df['ProductName'], products_df['Value']))
weights = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(capacities.keys()) != set(cabinets):
    raise ValueError('Mismatch in cabinet identifiers between capacity data and cabinets list.')
if set(values.keys()) != set(products) or set(weights.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers between value/weight data and products list.')
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinets, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[c, p] for c in cabinets for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[p] * x_vars[c, p] for p in products)) <= capacities[c] for c in cabinets), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')