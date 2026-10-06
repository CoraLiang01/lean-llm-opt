import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            products = pd.read_csv(product_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {product_path} with tried encodings.')
    for enc in encodings:
        try:
            capacities = pd.read_csv(capacity_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products.columns:
            raise ValueError(f"Missing column '{col}' in products.csv")
    for col in ['Warehouse ID', 'Capacity']:
        if col not in capacities.columns:
            raise ValueError(f"Missing column '{col}' in capacity.csv")
    vehicle_types = products['ProductName'].tolist()
    warehouses = capacities['Warehouse ID'].tolist()
    value = products.set_index('ProductName')['Value'].to_dict()
    weight = products.set_index('ProductName')['Weight'].to_dict()
    capacity = capacities.set_index('Warehouse ID')['Capacity'].to_dict()
    for i in vehicle_types:
        if i not in value or i not in weight:
            raise ValueError(f"Missing value or weight for vehicle type '{i}'")
    for j in warehouses:
        if j not in capacity:
            raise ValueError(f"Missing capacity for warehouse '{j}'")
    m = gp.Model('NewCarSalesInNorway2')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in vehicle_types for j in warehouses]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[i] * x[i, j] for (i, j) in keys)), GRB.MAXIMIZE)
    for j in warehouses:
        m.addConstr(gp.quicksum((weight[i] * x[i, j] for i in vehicle_types)) <= capacity[j], name=f'cap_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()