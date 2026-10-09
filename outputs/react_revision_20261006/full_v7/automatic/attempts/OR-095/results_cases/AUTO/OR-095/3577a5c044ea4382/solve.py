import gurobipy as gp
import pandas as pd
import numpy as np
import re

def match_col(df, pattern):
    pat = re.compile(pattern, re.IGNORECASE)
    for col in df.columns:
        if pat.fullmatch(col.strip()):
            return col
    raise KeyError(f"Column matching '{pattern}' not found in {df.columns.tolist()}")
product_resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
prod_df = pd.read_csv(product_resources_path, sep=',', dtype=str, keep_default_na=False)
prod_df.columns = [col.strip() for col in prod_df.columns]
widget_ids = [f'Widget{i}' for i in range(1, 142)]
prod_df['Product_norm'] = prod_df['Product'].str.strip()
missing_widgets = set(widget_ids) - set(prod_df['Product_norm'])
if missing_widgets:
    raise ValueError(f'Missing widget(s) in product_resources.csv: {sorted(missing_widgets)}')
prod_df = prod_df.set_index('Product_norm')
prod_df = prod_df.loc[widget_ids]
for col in ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']:
    prod_df[col] = pd.to_numeric(prod_df[col], errors='raise')
labor_hours = prod_df['LaborHours'].to_dict()
material_a = prod_df['MaterialA'].to_dict()
material_b = prod_df['MaterialB'].to_dict()
profit = prod_df['Profit'].to_dict()
res_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
res_df.columns = [col.strip() for col in res_df.columns]
res_df['Resource_norm'] = res_df['Resource'].str.strip().str.casefold()
res_limits = {}
for (_, row) in res_df.iterrows():
    res_limits[row['Resource_norm']] = int(row['MonthlyLimit'])
labor_limit = res_limits.get('laborhours')
materiala_limit = res_limits.get('materiala')
materialb_limit = res_limits.get('materialb')
if labor_limit is None or materiala_limit is None or materialb_limit is None:
    raise ValueError('Missing resource limits for LaborHours, MaterialA, or MaterialB.')
catalystx_per_widget3 = 5.0
catalystx_sale_price = 300.0
catalystx_disposal_cost = 200.0
catalystx_sales_cap = 1500.0
m = gp.Model('AerospaceWidgetProduction')
quantity_vars = m.addVars(widget_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
s_var = m.addVar(lb=0.0, ub=catalystx_sales_cap, vtype=gp.GRB.CONTINUOUS, name='s')
d_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='d')
obj = gp.quicksum((profit[w] * quantity_vars[w] for w in widget_ids)) + catalystx_sale_price * s_var - catalystx_disposal_cost * d_var
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((labor_hours[w] * quantity_vars[w] for w in widget_ids)) <= labor_limit, name='labor')
m.addConstr(gp.quicksum((material_a[w] * quantity_vars[w] for w in widget_ids)) <= materiala_limit, name='matA')
m.addConstr(gp.quicksum((material_b[w] * quantity_vars[w] for w in widget_ids)) <= materialb_limit, name='matB')
m.addConstr(catalystx_per_widget3 * quantity_vars['Widget3'] == s_var + d_var, name='catalystx_balance')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for w in widget_ids:
        print(f'{quantity_vars[w].VarName} {quantity_vars[w].X}')
    print(f'{s_var.VarName} {s_var.X}')
    print(f'{d_var.VarName} {d_var.X}')
else:
    print(f'Solver status: {m.status}')