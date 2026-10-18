import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', dtype=str, keep_default_na=False)
try:
    capacity = int(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError('Could not parse capacity from capacity.csv') from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
product_names = products_df['ProductName'].tolist()
try:
    values = {row['ProductName']: int(row['Value']) for _, row in products_df.iterrows()}
    weights = {row['ProductName']: int(row['Weight']) for _, row in products_df.iterrows()}
except Exception as e:
    raise ValueError('Could not parse Value or Weight as integer in products.csv') from e
if set(product_names) != set(values.keys()) or set(product_names) != set(weights.keys()):
    raise ValueError('Mismatch in product identifiers between names, values, and weights.')
m = gp.Model('PharmacyDrugOrder')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in product_names)) <= capacity, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')