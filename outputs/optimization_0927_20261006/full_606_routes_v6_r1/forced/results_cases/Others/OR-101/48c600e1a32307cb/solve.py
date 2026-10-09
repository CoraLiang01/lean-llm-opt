import gurobipy as gp
import pandas as pd
import numpy as np
device_time_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv'
monthly_device_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv'
unit_product_profits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv'
device_time_df = pd.read_csv(device_time_path, dtype=str, keep_default_na=False)
monthly_device_capacity_df = pd.read_csv(monthly_device_capacity_path, dtype=str, keep_default_na=False)
unit_product_profits_df = pd.read_csv(unit_product_profits_path, dtype=str, keep_default_na=False)
devices = device_time_df['Device'].str.strip().tolist()
devices_capacity = monthly_device_capacity_df['Device'].str.strip().tolist()
if set(devices) != set(devices_capacity):
    raise ValueError(f'Device sets in device_time.csv and monthly_device_capacity.csv do not match: {set(devices)} vs {set(devices_capacity)}')
product_cols = [col for col in device_time_df.columns if col != 'Device']
products = [p.strip() for p in product_cols]
products_profit = unit_product_profits_df['Product'].str.strip().tolist()
if set(products) != set(products_profit):
    raise ValueError(f'Product sets in device_time.csv and unit_product_profits.csv do not match: {set(products)} vs {set(products_profit)}')
unit_profit = {}
for (_, row) in unit_product_profits_df.iterrows():
    pid = row['Product'].strip()
    try:
        unit_profit[pid] = float(row['Unit_Profit'])
    except Exception as e:
        raise ValueError(f"Invalid Unit_Profit for product {pid}: {row['Unit_Profit']}") from e
monthly_capacity = {}
for (_, row) in monthly_device_capacity_df.iterrows():
    did = row['Device'].strip()
    try:
        monthly_capacity[did] = float(row['Monthly_Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid Monthly_Capacity for device {did}: {row['Monthly_Capacity']}") from e
device_time = {}
for (_, row) in device_time_df.iterrows():
    did = row['Device'].strip()
    for p in products:
        val = row[p]
        try:
            device_time[did, p] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid device_time for device {did}, product {p}: {val}') from e
for d in devices:
    if d not in monthly_capacity:
        raise ValueError(f'Device {d} missing in monthly_device_capacity.csv')
for p in products:
    if p not in unit_profit:
        raise ValueError(f'Product {p} missing in unit_product_profits.csv')
for d in devices:
    for p in products:
        if (d, p) not in device_time:
            raise ValueError(f'Missing device_time for device {d}, product {p}')
m = gp.Model('MonthlyProductionMaxProfit')
x_vars = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((unit_profit[p] * x_vars[p] for p in products)), gp.GRB.MAXIMIZE)
for d in devices:
    m.addConstr(gp.quicksum((device_time[d, p] * x_vars[p] for p in products)) <= monthly_capacity[d], name=f'DeviceCap_{d}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Production Plan (nonzero only) ---')
    for p in products:
        val = x_vars[p].X
        if val > 1e-06:
            print(f'  {p}: {val:.4f}')
    print('--- Device Utilization ---')
    for d in devices:
        used = sum((device_time[d, p] * x_vars[p].X for p in products))
        print(f'  {d}: Used {used:.2f} / Capacity {monthly_capacity[d]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')