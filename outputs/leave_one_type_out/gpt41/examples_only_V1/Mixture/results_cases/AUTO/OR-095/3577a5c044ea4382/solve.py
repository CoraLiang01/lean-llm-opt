import gurobipy as gp
import pandas as pd
import numpy as np
import re
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')

def norm_resource(s):
    return re.sub('\\s+', '', str(s)).casefold()
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(norm_resource)
resource_limits = {}
for _, row in resource_limits_df.iterrows():
    resource_limits[row['Resource_norm']] = float(row['MonthlyLimit'])
labor_key = norm_resource('LaborHours')
matA_key = norm_resource('MaterialA')
matB_key = norm_resource('MaterialB')
for k in [labor_key, matA_key, matB_key]:
    if k not in resource_limits:
        raise ValueError(f"Resource '{k}' not found in resource_limits.csv")
labor_limit = resource_limits[labor_key]
matA_limit = resource_limits[matA_key]
matB_limit = resource_limits[matB_key]
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
prod_df = pd.read_csv(product_resources_path, sep=',')
prod_df['Product'] = prod_df['Product'].astype(str).str.strip()
products = [f'Widget{i}' for i in range(1, 142)]
missing_widgets = [w for w in products if w not in set(prod_df['Product'])]
if missing_widgets:
    raise ValueError(f'Missing widgets in product_resources.csv: {missing_widgets}')
labor_hours = prod_df.set_index('Product')['LaborHours'].astype(float).to_dict()
material_a = prod_df.set_index('Product')['MaterialA'].astype(float).to_dict()
material_b = prod_df.set_index('Product')['MaterialB'].astype(float).to_dict()
profit = prod_df.set_index('Product')['Profit'].astype(float).to_dict()
m = gp.Model('WidgetProduction')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((profit[i] * x[i] for i in products)) + 300.0 * s - 200.0 * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in products)) <= labor_limit, name='labor')
m.addConstr(gp.quicksum((material_a[i] * x[i] for i in products)) <= matA_limit, name='matA')
m.addConstr(gp.quicksum((material_b[i] * x[i] for i in products)) <= matB_limit, name='matB')
if 'Widget3' not in products:
    raise ValueError('Widget3 not found in product_resources.csv')
m.addConstr(5.0 * x['Widget3'] == s + d, name='catalyst_balance')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\n--- Production Plan ---')
    for i in products:
        val = x[i].X
        if val > 1e-06:
            print(f'{i:8s}: {val:10.2f} units')
    print('\n--- CatalystX Handling ---')
    print(f'Sold:      {s.X:.2f} kg')
    print(f'Disposed:  {d.X:.2f} kg')
    print(f"Total generated (should equal 5*Widget3): {5.0 * x['Widget3'].X:.2f} kg")
else:
    print(f'No optimal solution found. Status: {m.status}')