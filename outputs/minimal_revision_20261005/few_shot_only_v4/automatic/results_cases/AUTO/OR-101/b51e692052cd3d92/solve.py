import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    unit_profit_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv'
    device_time_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv'
    device_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv'

    def read_csv_with_encoding(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    df_profit = read_csv_with_encoding(unit_profit_path)
    df_time = read_csv_with_encoding(device_time_path)
    df_capacity = read_csv_with_encoding(device_capacity_path)
    products = df_profit['Product'].astype(str).tolist()
    devices = df_time['Device'].astype(str).tolist()
    devices_capacity = df_capacity['Device'].astype(str).tolist()
    if set(devices) != set(devices_capacity):
        raise ValueError('Device lists in device_time.csv and monthly_device_capacity.csv do not match.')
    profit = {}
    for (_, row) in df_profit.iterrows():
        prod = str(row['Product'])
        if prod in profit:
            raise ValueError(f'Duplicate product in unit_product_profits.csv: {prod}')
        profit[prod] = float(row['Unit_Profit'])
    a = {}
    for (_, row) in df_time.iterrows():
        dev = str(row['Device'])
        a[dev] = {}
        for prod in products:
            if prod not in row:
                raise ValueError(f'Product {prod} missing in device_time.csv columns.')
            a[dev][prod] = float(row[prod])
    capacity = {}
    for (_, row) in df_capacity.iterrows():
        dev = str(row['Device'])
        if dev in capacity:
            raise ValueError(f'Duplicate device in monthly_device_capacity.csv: {dev}')
        capacity[dev] = float(row['Monthly_Capacity'])
    for dev in devices:
        if dev not in a or dev not in capacity:
            raise ValueError(f'Device {dev} missing in data.')
        for prod in products:
            if prod not in a[dev]:
                raise ValueError(f'Product {prod} missing for device {dev} in device_time.csv.')
    m = gp.Model('Monthly_Production_Optimization')
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((profit[i] * x[i] for i in products)), GRB.MAXIMIZE)
    for k in devices:
        m.addConstr(gp.quicksum((a[k][i] * x[i] for i in products)) <= capacity[k], name=f'cap_{k}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()