import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in decode_errors:
        try:
            products_df = pd.read_csv(product_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == decode_errors[-1]:
                raise
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
    for enc in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == decode_errors[-1]:
                raise
    required_product_cols = {'ProductName', 'Weight', 'Value'}
    if not required_product_cols.issubset(products_df.columns):
        missing = required_product_cols - set(products_df.columns)
        raise ValueError(f'Missing columns in products.csv: {missing}')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    I = products_df['ProductName'].tolist()
    if len(I) != len(set(I)):
        raise ValueError('Duplicate ProductName entries found in products.csv')
    try:
        w_i = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
        v_i = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Weight/Value to float: {e}')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    for i in I:
        if i not in w_i or i not in v_i:
            raise ValueError(f'Missing weight or value for product {i}')
    m = gp.Model('Supermarket_Produce_Order')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w_i[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()