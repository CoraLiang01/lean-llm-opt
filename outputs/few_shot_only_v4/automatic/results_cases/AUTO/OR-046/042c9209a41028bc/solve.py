import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise ValueError("Missing 'Capacity' column in capacity.csv")
try:
    total_capacity = float(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError(f'Could not parse capacity value: {e}')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv', dtype=str, keep_default_na=False)
required_columns = {'Product Name', 'Weight', 'Value'}
if not required_columns.issubset(products_df.columns):
    raise ValueError(f'products.csv missing columns: {required_columns - set(products_df.columns)}')
product_names = products_df['Product Name'].tolist()
weights = {}
values = {}
for idx, row in products_df.iterrows():
    pname = row['Product Name']
    try:
        weights[pname] = float(row['Weight'])
        values[pname] = float(row['Value'])
    except Exception as e:
        raise ValueError(f"Error parsing numeric fields for product '{pname}': {e}")
if set(product_names) != set(weights.keys()) or set(product_names) != set(values.keys()):
    raise ValueError('Mismatch in product identifiers between names, weights, and values.')
m = gp.Model('Supermarket_Replenishment')
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in product_names)) <= total_capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')