import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(x):
    if isinstance(x, str):
        return re.sub('\\s+', '', x).strip()
    return x
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].apply(lambda x: normalize_id(str(x)).casefold())
resource_limits = dict(zip(resource_limits_df['Resource_norm'], resource_limits_df['MonthlyLimit']))
labor_key = normalize_id('LaborHours').casefold()
materiala_key = normalize_id('MaterialA').casefold()
materialb_key = normalize_id('MaterialB').casefold()
if labor_key not in resource_limits or materiala_key not in resource_limits or materialb_key not in resource_limits:
    raise ValueError('Missing required resource limits in resource_limits.csv')
labor_limit = float(resource_limits[labor_key])
materiala_limit = float(resource_limits[materiala_key])
materialb_limit = float(resource_limits[materialb_key])
product_resources_df = pd.read_csv(product_resources_path, sep=',')
product_resources_df['Product_norm'] = product_resources_df['Product'].apply(lambda x: normalize_id(str(x)))
products = list(product_resources_df['Product_norm'])
product_norm_to_orig = dict(zip(product_resources_df['Product_norm'], product_resources_df['Product']))
labor_hours = dict(zip(product_resources_df['Product_norm'], product_resources_df['LaborHours']))
material_a = dict(zip(product_resources_df['Product_norm'], product_resources_df['MaterialA']))
material_b = dict(zip(product_resources_df['Product_norm'], product_resources_df['MaterialB']))
profit = dict(zip(product_resources_df['Product_norm'], product_resources_df['Profit']))
widget3_norm = normalize_id('Widget3')
if widget3_norm not in products:
    raise ValueError('Widget3 not found in product_resources.csv')

def solve_widget_production():
    m = gp.Model('AerospaceWidgetProduction')
    x = m.addVars(products, name='x', lb=0.0)
    s = m.addVar(name='CatalystX_sold', lb=0.0)
    d = m.addVar(name='CatalystX_disposed', lb=0.0)
    obj = gp.quicksum((profit[i] * x[i] for i in products)) + 300 * s - 200 * d
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((labor_hours[i] * x[i] for i in products)) <= labor_limit, name='LaborHours_limit')
    m.addConstr(gp.quicksum((material_a[i] * x[i] for i in products)) <= materiala_limit, name='MaterialA_limit')
    m.addConstr(gp.quicksum((material_b[i] * x[i] for i in products)) <= materialb_limit, name='MaterialB_limit')
    m.addConstr(5.0 * x[widget3_norm] == s + d, name='CatalystX_balance')
    m.addConstr(s <= 1500.0, name='CatalystX_sales_cap')
    m.optimize()
    return m
m = solve_widget_production()