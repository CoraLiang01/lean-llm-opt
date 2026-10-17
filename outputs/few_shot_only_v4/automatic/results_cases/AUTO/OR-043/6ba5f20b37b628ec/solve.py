import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv', dtype=str, keep_default_na=False)
for col in capacity_df.columns:
    if col != capacity_df.columns[0]:
        capacity_df[col] = pd.to_numeric(capacity_df[col], errors='raise')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv', dtype=str, keep_default_na=False)
for col in products_df.columns:
    if col != products_df.columns[0]:
        products_df[col] = pd.to_numeric(products_df[col], errors='raise')
product_id_col = products_df.columns[0]
value_col = [c for c in products_df.columns if 'value' in c.lower()][0]
weight_col = [c for c in products_df.columns if 'weight' in c.lower()][0]
products = products_df[product_id_col].tolist()
values = dict(zip(products_df[product_id_col], products_df[value_col]))
weights = dict(zip(products_df[product_id_col], products_df[weight_col]))
resource_id_col = capacity_df.columns[0]
resources = capacity_df[resource_id_col].tolist()
capacity_cols = [c for c in capacity_df.columns if c != resource_id_col]
capacities = {}
for idx, row in capacity_df.iterrows():
    resource = row[resource_id_col]
    for cap_col in capacity_cols:
        capacities[resource, cap_col] = row[cap_col]
if len(resources) == 1 and len(capacity_cols) == 1:
    total_capacity = capacities[resources[0], capacity_cols[0]]
    resource_constraints = [(resources[0], capacity_cols[0], total_capacity)]
else:
    resource_constraints = []
    for (resource, cap_col), cap_val in capacities.items():
        resource_constraints.append((resource, cap_col, cap_val))
m = gp.Model('Pharmacy_Inventory_Optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
if len(resource_constraints) == 1:
    m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in products)) <= resource_constraints[0][2], name='stock_capacity')
else:
    for resource, cap_col, cap_val in resource_constraints:
        m.addConstr(gp.quicksum((weights[p] * x_vars[p] for p in products)) <= cap_val, name=f'cap_{resource}_{cap_col}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')