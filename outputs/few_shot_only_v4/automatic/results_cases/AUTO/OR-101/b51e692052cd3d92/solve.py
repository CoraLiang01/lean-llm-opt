import gurobipy as gp
from gurobipy import GRB
import pandas as pd
device_time_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv'
monthly_device_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv'
unit_product_profits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv'
profits_df = pd.read_csv(unit_product_profits_path, dtype=str, keep_default_na=False)
if 'Product' not in profits_df.columns or 'Unit_Profit' not in profits_df.columns:
    raise ValueError('unit_product_profits.csv must have columns Product and Unit_Profit')
products = list(profits_df['Product'])
profit = {}
for _, row in profits_df.iterrows():
    try:
        profit[row['Product']] = float(row['Unit_Profit'])
    except Exception:
        raise ValueError(f"Invalid Unit_Profit for product {row['Product']}")
capacity_df = pd.read_csv(monthly_device_capacity_path, dtype=str, keep_default_na=False)
if 'Device' not in capacity_df.columns or 'Monthly_Capacity' not in capacity_df.columns:
    raise ValueError('monthly_device_capacity.csv must have columns Device and Monthly_Capacity')
devices = list(capacity_df['Device'])
capacity = {}
for _, row in capacity_df.iterrows():
    try:
        capacity[row['Device']] = float(row['Monthly_Capacity'])
    except Exception:
        raise ValueError(f"Invalid Monthly_Capacity for device {row['Device']}")
device_time_df = pd.read_csv(device_time_path, dtype=str, keep_default_na=False)
if 'Device' not in device_time_df.columns:
    raise ValueError('device_time.csv must have a Device column')
if not all((p in device_time_df.columns for p in products)):
    raise ValueError('device_time.csv must have a column for every product')
a = {}
for _, row in device_time_df.iterrows():
    device = row['Device']
    if device not in devices:
        continue
    a[device] = {}
    for product in products:
        try:
            a[device][product] = float(row[product])
        except Exception:
            raise ValueError(f'Invalid processing time for device {device}, product {product}')
if set(a.keys()) != set(devices):
    raise ValueError('Mismatch between devices in device_time.csv and monthly_device_capacity.csv')
for device in devices:
    if set(a[device].keys()) != set(products):
        raise ValueError(f'Mismatch between products in device_time.csv and unit_product_profits.csv for device {device}')
m = gp.Model('Monthly_Production_Optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((profit[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((a[d][p] * x_vars[p] for p in products)) <= capacity[d] for d in devices), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')