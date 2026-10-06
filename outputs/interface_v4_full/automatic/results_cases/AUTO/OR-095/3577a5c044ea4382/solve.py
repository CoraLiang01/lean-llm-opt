import gurobipy as gp
import pandas as pd
import numpy as np
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold()

def get_limit(resource_name):
    norm = resource_name.strip().casefold()
    matches = resource_limits_df[resource_limits_df['Resource_norm'] == norm]
    if len(matches) != 1:
        raise ValueError(f"Resource '{resource_name}' not found or not unique in resource_limits.csv")
    return float(matches['MonthlyLimit'].iloc[0])
labor_limit = get_limit('LaborHours')
materialA_limit = get_limit('MaterialA')
materialB_limit = get_limit('MaterialB')
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_df = pd.read_csv(product_resources_path, sep=',')
product_df['Product_norm'] = product_df['Product'].str.strip()
products = [f'Widget{i}' for i in range(1, 142)]
missing_products = [p for p in products if p not in set(product_df['Product_norm'])]
if missing_products:
    raise ValueError(f'Missing products in product_resources.csv: {missing_products}')
labor_hours = {}
materialA = {}
materialB = {}
profit = {}
for _, row in product_df.iterrows():
    prod = row['Product'].strip()
    if prod in products:
        labor_hours[prod] = float(row['LaborHours'])
        materialA[prod] = float(row['MaterialA'])
        materialB[prod] = float(row['MaterialB'])
        profit[prod] = float(row['Profit'])
m = gp.Model('AerospaceWidgetProduction')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, ub=1500.0, vtype=gp.GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((profit[i] * x[i] for i in products)) + 300.0 * s - 200.0 * d
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in products)) <= labor_limit, name='LaborHours')
m.addConstr(gp.quicksum((materialA[i] * x[i] for i in products)) <= materialA_limit, name='MaterialA')
m.addConstr(gp.quicksum((materialB[i] * x[i] for i in products)) <= materialB_limit, name='MaterialB')
widget3_name = 'Widget3'
if widget3_name not in products:
    raise ValueError('Widget3 not found in product list.')
m.addConstr(5.0 * x[widget3_name] == s + d, name='CatalystX_Balance')
m.optimize()