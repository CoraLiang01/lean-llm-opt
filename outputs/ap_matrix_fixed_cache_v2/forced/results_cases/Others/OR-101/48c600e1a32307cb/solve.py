import gurobipy as gp
import pandas as pd
import numpy as np
device_time_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv'
monthly_device_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv'
unit_product_profits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv'
device_time_df = pd.read_csv(device_time_path, dtype=str, keep_default_na=False)
monthly_device_capacity_df = pd.read_csv(monthly_device_capacity_path, dtype=str, keep_default_na=False)
unit_product_profits_df = pd.read_csv(unit_product_profits_path, dtype=str, keep_default_na=False)
products = [f'P{i}' for i in range(1, 112)]
devices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
if not set(products).issubset(set(unit_product_profits_df['Product'].str.strip())):
    missing = set(products) - set(unit_product_profits_df['Product'].str.strip())
    raise ValueError(f'Missing unit profit for products: {missing}')
unit_profit = unit_product_profits_df.set_index(unit_product_profits_df['Product'].str.strip())['Unit_Profit'].astype(float).to_dict()
unit_profit = {p: unit_profit[p] for p in products}
if not set(devices).issubset(set(device_time_df['Device'].str.strip())):
    missing = set(devices) - set(device_time_df['Device'].str.strip())
    raise ValueError(f'Missing device_time rows for devices: {missing}')
if not set(products).issubset(set(device_time_df.columns[1:])):
    missing = set(products) - set(device_time_df.columns[1:])
    raise ValueError(f'Missing device_time columns for products: {missing}')
device_time = {}
for _, row in device_time_df.iterrows():
    device = row['Device'].strip()
    if device not in devices:
        continue
    device_time[device] = {}
    for p in products:
        val = row[p]
        try:
            device_time[device][p] = float(val)
        except Exception:
            raise ValueError(f"Invalid device_time for device {device}, product {p}: '{val}'")
if not set(devices).issubset(set(monthly_device_capacity_df['Device'].str.strip())):
    missing = set(devices) - set(monthly_device_capacity_df['Device'].str.strip())
    raise ValueError(f'Missing monthly capacity for devices: {missing}')
monthly_capacity = monthly_device_capacity_df.set_index(monthly_device_capacity_df['Device'].str.strip())['Monthly_Capacity'].astype(float).to_dict()
monthly_capacity = {d: monthly_capacity[d] for d in devices}
m = gp.Model('MonthlyProductionPlanning')
x_vars = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((unit_profit[p] * x_vars[p] for p in products)), gp.GRB.MAXIMIZE)
for d in devices:
    m.addConstr(gp.quicksum((device_time[d][p] * x_vars[p] for p in products)) <= monthly_capacity[d], name=f'cap_{d}')
m.optimize()