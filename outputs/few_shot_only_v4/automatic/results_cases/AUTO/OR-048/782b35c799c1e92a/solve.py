import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', dtype=str, keep_default_na=False)
if not {'StorageID', 'Capacity'}.issubset(capacity_df.columns):
    raise ValueError('Missing required columns in capacity.csv')
storage_ids = list(capacity_df['StorageID'])
capacities = {}
for idx, row in capacity_df.iterrows():
    try:
        capacities[row['StorageID']] = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity value for StorageID {row['StorageID']}")
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise ValueError('Missing required columns in products.csv')
product_names = list(products_df['ProductName'])
values = {}
weights = {}
for idx, row in products_df.iterrows():
    try:
        values[row['ProductName']] = int(row['Value'])
        weights[row['ProductName']] = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid value or weight for ProductName {row['ProductName']}")
if len(storage_ids) == 0 or len(product_names) == 0:
    raise ValueError('No storage areas or products found in input files')
if set(values.keys()) != set(product_names) or set(weights.keys()) != set(product_names):
    raise ValueError('Mismatch in product identifiers between value/weight and product list')
if set(capacities.keys()) != set(storage_ids):
    raise ValueError('Mismatch in storage identifiers between capacities and storage list')
m = gp.Model('Amazon_AC_Storage_Allocation')
x_vars = m.addVars(storage_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x_vars[i, j] for i in storage_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x_vars[i, j] for j in product_names)) <= capacities[i] for i in storage_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')