import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()
resource_limit_map = {}
for (_, row) in resource_limits_df.iterrows():
    key = row['Resource'].strip()
    key_norm = key.casefold()
    resource_limit_map[key_norm] = int(row['MonthlyLimit'])
required_resources = ['LaborHours', 'MaterialA', 'MaterialB']
required_resources_norm = [r.strip().casefold() for r in required_resources]
for r_norm in required_resources_norm:
    if r_norm not in resource_limit_map:
        raise ValueError(f"Missing resource limit for '{r_norm}' in resource_limits.csv")
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_resources_df = pd.read_csv(product_resources_path, sep=',')
product_resources_df['Product_norm'] = product_resources_df['Product'].str.strip()
product_ids = [f'Widget{i}' for i in range(1, 142)]
product_set = set(product_resources_df['Product_norm'])
missing_products = [pid for pid in product_ids if pid not in product_set]
if missing_products:
    raise ValueError(f'Missing products in product_resources.csv: {missing_products}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for (_, row) in product_resources_df.iterrows():
    pid = row['Product'].strip()
    if pid in product_ids:
        labor_hours[pid] = float(row['LaborHours'])
        material_a[pid] = int(row['MaterialA'])
        material_b[pid] = int(row['MaterialB'])
        profit[pid] = int(row['Profit'])
for pid in product_ids:
    if pid not in labor_hours or pid not in material_a or pid not in material_b or (pid not in profit):
        raise ValueError(f'Missing coefficients for product {pid}')
catalystx_producer = 'Widget3'
catalystx_byproduct_rate = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('AerospaceWidgetProduction')
x = m.addVars(product_ids, lb=0.0, vtype=GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='d')
m.addConstr(gp.quicksum((labor_hours[pid] * x[pid] for pid in product_ids)) <= resource_limit_map['laborhours'], name='labor_hours')
m.addConstr(gp.quicksum((material_a[pid] * x[pid] for pid in product_ids)) <= resource_limit_map['materiala'], name='material_a')
m.addConstr(gp.quicksum((material_b[pid] * x[pid] for pid in product_ids)) <= resource_limit_map['materialb'], name='material_b')
m.addConstr(catalystx_byproduct_rate * x[catalystx_producer] == s + d, name='catalystx_balance')
m.addConstr(s <= catalystx_sales_cap, name='catalystx_sales_cap')
objective = gp.quicksum((profit[pid] * x[pid] for pid in product_ids)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(objective, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for pid in product_ids:
        print(f'{x[pid].VarName} {x[pid].X}')
    print(f'{s.VarName} {s.X}')
    print(f'{d.VarName} {d.X}')
else:
    print(f'Solver status: {m.Status}')