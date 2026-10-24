import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
    df_cap = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    df_prod = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    df_cap['VehicleID'] = df_cap['VehicleID'].str.strip()
    df_cap['VehicleType'] = df_cap['VehicleType'].str.strip()
    df_cap['Capacity'] = df_cap['Capacity'].str.strip().astype(int)
    df_prod['ProductName'] = df_prod['ProductName'].str.strip()
    df_prod['Value'] = df_prod['Value'].str.strip().astype(int)
    capacity_dict = dict(zip(df_cap['VehicleID'], df_cap['Capacity']))
    value_by_name = dict(zip(df_prod['ProductName'], df_prod['Value']))
    vehicle_ids = df_cap['VehicleID'].tolist()
    value_dict = {}
    for (_, row) in df_cap.iterrows():
        vid = row['VehicleID']
        vtype = row['VehicleType']
        if vtype not in value_by_name:
            raise ValueError(f"Missing Value for VehicleType '{vtype}' corresponding to VehicleID '{vid}'")
        value_dict[vid] = value_by_name[vtype]
    total_capacity = sum(capacity_dict.values())
    m = gurobipy.Model('VehicleOrder')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(vehicle_ids, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((value_dict[i] * x[i] for i in vehicle_ids)), GRB.MAXIMIZE)
    for i in vehicle_ids:
        m.addConstr(x[i] <= capacity_dict[i], name=f'cap_{i}')
    m.addConstr(gurobipy.quicksum((x[i] for i in vehicle_ids)) <= total_capacity, name='total_cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()