import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
    df_cap = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'DisplayID' not in df_cap.columns or 'Capacity' not in df_cap.columns:
        raise ValueError('Capacity CSV missing required columns.')
    df_cap['Capacity'] = pd.to_numeric(df_cap['Capacity'], errors='raise')
    c_i = dict(zip(df_cap['DisplayID'], df_cap['Capacity']))
    df_prod = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(df_prod.columns):
        raise ValueError('Products CSV missing required columns.')
    df_prod['Value'] = pd.to_numeric(df_prod['Value'], errors='raise')
    df_prod['Weight'] = pd.to_numeric(df_prod['Weight'], errors='raise')
    v_j = dict(zip(df_prod['ProductName'], df_prod['Value']))
    w_j = dict(zip(df_prod['ProductName'], df_prod['Weight']))
    I = list(c_i.keys())
    J = list(v_j.keys())
    if not set(J).issubset(set(w_j.keys())):
        missing = set(J) - set(w_j.keys())
        raise ValueError(f'Missing weight data for products: {missing}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((v_j[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((w_j[j] * x[i, j] for j in J)) <= c_i[i], name=f'cap_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()