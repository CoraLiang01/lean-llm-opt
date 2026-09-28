import gurobipy as gp
import pandas as pd
import numpy as np
import re
device_time_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv'
monthly_device_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv'
unit_product_profits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv'
device_time_df = pd.read_csv(device_time_path, sep=',')
monthly_device_capacity_df = pd.read_csv(monthly_device_capacity_path, sep=',')
unit_product_profits_df = pd.read_csv(unit_product_profits_path, sep=',')
devices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
products = [f'P{i}' for i in range(1, 112)]
if not set(devices).issubset(set(device_time_df['Device'].astype(str))):
    missing = set(devices) - set(device_time_df['Device'].astype(str))
    raise ValueError(f'Missing device_time rows for devices: {missing}')
if not set(products).issubset(set(device_time_df.columns)):
    missing = set(products) - set(device_time_df.columns)
    raise ValueError(f'Missing device_time columns for products: {missing}')
if not set(devices).issubset(set(monthly_device_capacity_df['Device'].astype(str))):
    missing = set(devices) - set(monthly_device_capacity_df['Device'].astype(str))
    raise ValueError(f'Missing monthly_device_capacity rows for devices: {missing}')
if not set(products).issubset(set(unit_product_profits_df['Product'].astype(str))):
    missing = set(products) - set(unit_product_profits_df['Product'].astype(str))
    raise ValueError(f'Missing unit_product_profits rows for products: {missing}')
unit_profit = unit_product_profits_df.set_index('Product')['Unit_Profit'].astype(float).to_dict()
device_time = {}
for _, row in device_time_df.iterrows():
    dev = str(row['Device']).strip()
    if dev not in devices:
        continue
    device_time[dev] = {}
    for prod in products:
        device_time[dev][prod] = float(row[prod])
monthly_capacity = monthly_device_capacity_df.set_index('Device')['Monthly_Capacity'].astype(float).to_dict()
m = gp.Model('MonthlyProductionPlan')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((unit_profit[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
for j in devices:
    m.addConstr(gp.quicksum((device_time[j][i] * x[i] for i in products)) <= monthly_capacity[j], name=f'cap_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Production Plan ---')
    for i in products:
        val = x[i].X
        if val > 1e-06:
            print(f'{i}: {val:.4f}')
else:
    print(f'No optimal solution found. Status: {m.status}')