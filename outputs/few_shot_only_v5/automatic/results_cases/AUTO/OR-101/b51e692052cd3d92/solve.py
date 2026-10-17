import pandas as pd
import gurobipy as gp
from gurobipy import GRB
profits_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv', dtype=str, keep_default_na=False)
device_time_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv', dtype=str, keep_default_na=False)
products = [p.strip() for p in profits_df['Product']]
devices = [d.strip() for d in capacity_df['Device']]
profit = {}
for _, row in profits_df.iterrows():
    try:
        profit[row['Product'].strip()] = float(row['Unit_Profit'])
    except Exception as e:
        raise ValueError(f"Invalid profit for product {row['Product']}: {row['Unit_Profit']}") from e
capacity = {}
for _, row in capacity_df.iterrows():
    try:
        capacity[row['Device'].strip()] = float(row['Monthly_Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid capacity for device {row['Device']}: {row['Monthly_Capacity']}") from e
device_time = {}
device_time_df = device_time_df.set_index('Device')
for d in devices:
    if d not in device_time_df.index:
        raise ValueError(f'Device {d} missing from device_time.csv')
    row = device_time_df.loc[d]
    device_time[d] = {}
    for p in products:
        val = row[p].strip()
        try:
            device_time[d][p] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid processing time for device {d}, product {p}: {val}') from e
if set(profit.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers between profits and products list')
if set(capacity.keys()) != set(devices):
    raise ValueError('Mismatch in device identifiers between capacity and devices list')
for d in devices:
    if set(device_time[d].keys()) != set(products):
        raise ValueError(f'Mismatch in product identifiers for device {d} in device_time')
m = gp.Model('Mixture18_Production')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((profit[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
for d in devices:
    m.addConstr(gp.quicksum((device_time[d][p] * x_vars[p] for p in products)) <= capacity[d], name=f'device_{d}_capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')