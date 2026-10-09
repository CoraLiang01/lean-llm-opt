import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    device_time_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv'
    monthly_device_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv'
    unit_product_profits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv'

    def read_csv_fallback(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    device_time_df = read_csv_fallback(device_time_path)
    monthly_device_capacity_df = read_csv_fallback(monthly_device_capacity_path)
    unit_product_profits_df = read_csv_fallback(unit_product_profits_path)
    products = list(unit_product_profits_df['Product'])
    devices_time = list(device_time_df['Device'])
    devices_capacity = list(monthly_device_capacity_df['Device'])
    devices = sorted(set(devices_time) | set(devices_capacity))
    device_time_products = [col for col in device_time_df.columns if col != 'Device']
    missing_products = set(products) - set(device_time_products)
    if missing_products:
        raise ValueError(f'Products missing in device_time.csv: {missing_products}')
    missing_time_devices = set(devices) - set(devices_time)
    missing_capacity_devices = set(devices) - set(devices_capacity)
    if missing_time_devices:
        raise ValueError(f'Devices missing in device_time.csv: {missing_time_devices}')
    if missing_capacity_devices:
        raise ValueError(f'Devices missing in monthly_device_capacity.csv: {missing_capacity_devices}')
    try:
        p_i = {row['Product']: float(row['Unit_Profit']) for (_, row) in unit_product_profits_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing Unit_Profit: {e}')
    t_di = {}
    for (_, row) in device_time_df.iterrows():
        d = row['Device']
        for i in products:
            try:
                val = row[i]
                t_di[d, i] = float(val)
            except Exception as e:
                raise ValueError(f'Error parsing processing time for device {d}, product {i}: {e}')
    try:
        c_d = {row['Device']: float(row['Monthly_Capacity']) for (_, row) in monthly_device_capacity_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing Monthly_Capacity: {e}')
    for d in devices:
        if d not in c_d:
            raise ValueError(f'Missing capacity for device {d}')
        for i in products:
            if (d, i) not in t_di:
                raise ValueError(f'Missing processing time for device {d}, product {i}')
    for i in products:
        if i not in p_i:
            raise ValueError(f'Missing unit profit for product {i}')
    m = gp.Model('monthly_production_plan')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((p_i[i] * quantity_vars[i] for i in products)), GRB.MAXIMIZE)
    for d in devices:
        m.addConstr(gp.quicksum((t_di[d, i] * quantity_vars[i] for i in products)) <= c_d[d], name=f'cap_{d}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')