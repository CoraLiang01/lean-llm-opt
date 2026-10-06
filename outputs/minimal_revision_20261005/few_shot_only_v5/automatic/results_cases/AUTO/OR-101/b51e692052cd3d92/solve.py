import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    device_time_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv'
    monthly_device_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv'
    unit_product_profits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv'
    device_time_df = read_csv_robust(device_time_path)
    device_capacity_df = read_csv_robust(monthly_device_capacity_path)
    product_profit_df = read_csv_robust(unit_product_profits_path)
    devices = device_time_df['Device'].astype(str).tolist()
    products = [col for col in device_time_df.columns if col != 'Device']
    profit_products = product_profit_df['Product'].astype(str).tolist()
    if set(products) != set(profit_products):
        raise ValueError('Mismatch between products in device_time.csv and unit_product_profits.csv')
    capacity_devices = device_capacity_df['Device'].astype(str).tolist()
    if set(devices) != set(capacity_devices):
        raise ValueError('Mismatch between devices in device_time.csv and monthly_device_capacity.csv')
    profit = dict(zip(product_profit_df['Product'].astype(str), product_profit_df['Unit_Profit']))
    device_capacity = dict(zip(device_capacity_df['Device'].astype(str), device_capacity_df['Monthly_Capacity']))
    device_time = {}
    for (_, row) in device_time_df.iterrows():
        d = str(row['Device'])
        for p in products:
            device_time[d, p] = row[p]
    m = gp.Model('FactoryProduction')
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((profit[p] * x[p] for p in products)), GRB.MAXIMIZE)
    for d in devices:
        m.addConstr(gp.quicksum((device_time[d, p] * x[p] for p in products)) <= device_capacity[d], name=f'device_cap_{d}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()