import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
    capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in capacity_df.columns:
        raise ValueError('Capacity column missing in capacity file')
    if capacity_df.empty or capacity_df['Capacity'].iloc[0] == '':
        raise ValueError('Capacity value missing')
    try:
        capacity = int(capacity_df['Capacity'].iloc[0])
    except ValueError:
        raise ValueError('Capacity value is not an integer')
    products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(products_df.columns):
        missing = required_cols - set(products_df.columns)
        raise ValueError(f'Missing columns in products file: {missing}')
    product_names = []
    values = {}
    weights = {}
    for (_, row) in products_df.iterrows():
        pname = row['ProductName']
        val_str = row['Value']
        wt_str = row['Weight']
        if pname == '' or val_str == '' or wt_str == '':
            raise ValueError(f'Empty value in product row: {row.to_dict()}')
        try:
            val = float(val_str)
            wt = float(wt_str)
        except ValueError:
            raise ValueError(f'Non-numeric value in product row: {row.to_dict()}')
        product_names.append(pname)
        values[pname] = val
        weights[pname] = wt
    m = gurobipy.Model('PharmaInventory')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((values[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((weights[i] * x[i] for i in product_names)) <= capacity, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()