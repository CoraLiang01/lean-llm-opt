import pandas as pd
import numpy as np
from gurobipy import Model, GRB
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
resource_limits_df = pd.read_csv(resource_limits_path, sep=',')
resource_limits_df['Resource_norm'] = resource_limits_df['Resource'].str.strip().str.casefold().str.replace('\\s+', '', regex=True)
resource_limits = {}
for (_, row) in resource_limits_df.iterrows():
    resource_limits[row['Resource_norm']] = int(row['MonthlyLimit'])
resource_map = {'laborhours': 'LaborHours', 'materiala': 'MaterialA', 'materialb': 'MaterialB'}
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
product_df = pd.read_csv(product_resources_path, sep=',')
product_df['Product_norm'] = product_df['Product'].str.strip()
widget_names = [f'Widget{i}' for i in range(1, 142)]
missing_widgets = set(widget_names) - set(product_df['Product_norm'])
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {missing_widgets}')
labor_hours = {}
material_a = {}
material_b = {}
profit = {}
for (_, row) in product_df.iterrows():
    prod = row['Product_norm']
    if prod in widget_names:
        labor_hours[prod] = float(row['LaborHours'])
        material_a[prod] = float(row['MaterialA'])
        material_b[prod] = float(row['MaterialB'])
        profit[prod] = float(row['Profit'])
widget3_name = 'Widget3'
if widget3_name not in widget_names:
    raise ValueError('Widget3 not found in product_resources.csv')
catalystx_per_widget3 = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = Model('AerospaceWidgetProduction')
x = m.addVars(widget_names, lb=0.0, vtype=GRB.CONTINUOUS, name='')
s = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='s')
d = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='d')
obj = sum((profit[i] * x[i] for i in widget_names)) + catalystx_sale_price * s - catalystx_disposal_cost * d
m.setObjective(obj, GRB.MAXIMIZE)
labor_limit = resource_limits['laborhours']
m.addConstr(sum((labor_hours[i] * x[i] for i in widget_names)) <= labor_limit, name='LaborHoursLimit')
materiala_limit = resource_limits['materiala']
m.addConstr(sum((material_a[i] * x[i] for i in widget_names)) <= materiala_limit, name='MaterialALimit')
materialb_limit = resource_limits['materialb']
m.addConstr(sum((material_b[i] * x[i] for i in widget_names)) <= materialb_limit, name='MaterialBLimit')
m.addConstr(catalystx_per_widget3 * x[widget3_name] == s + d, name='CatalystXBalance')
m.addConstr(s <= catalystx_sales_cap, name='CatalystXSalesCap')
m.optimize()