import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path_limits = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv'
    path_products = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv'
    df_limits = pd.read_csv(path_limits, dtype=str, keep_default_na=False)
    df_products = pd.read_csv(path_products, dtype=str, keep_default_na=False)
    numeric_cols = ['LaborHours', 'MaterialA', 'MaterialB', 'Profit']
    for col in numeric_cols:
        df_products[col] = df_products[col].astype(float)
    df_limits['MonthlyLimit'] = df_limits['MonthlyLimit'].astype(float)

    def get_limit(resource_name):
        row = df_limits[df_limits['Resource'] == resource_name]
        if row.empty:
            raise ValueError(f'Missing resource limit for {resource_name}')
        return row['MonthlyLimit'].iloc[0]
    R_L = get_limit('LaborHours')
    R_A = get_limit('MaterialA')
    R_B = get_limit('MaterialB')
    c_sell = 300.0
    c_dispose = 200.0
    S_max = 1500.0
    k = 5.0
    product_names = df_products['Product'].tolist()
    if 'Widget3' not in product_names:
        raise ValueError('Product Widget3 not found in product list')
    L = dict(zip(product_names, df_products['LaborHours']))
    A = dict(zip(product_names, df_products['MaterialA']))
    B = dict(zip(product_names, df_products['MaterialB']))
    pi = dict(zip(product_names, df_products['Profit']))
    m = gurobipy.Model('WidgetProduction')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.CONTINUOUS, name='')
    s = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='s')
    d = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d')
    obj = gurobipy.quicksum((pi[p] * x[p] for p in product_names)) + c_sell * s - c_dispose * d
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((L[p] * x[p] for p in product_names)) <= R_L, name='LaborLimit')
    m.addConstr(gurobipy.quicksum((A[p] * x[p] for p in product_names)) <= R_A, name='MaterialALimit')
    m.addConstr(gurobipy.quicksum((B[p] * x[p] for p in product_names)) <= R_B, name='MaterialBLimit')
    m.addConstr(k * x['Widget3'] == s + d, name='CatalystBalance')
    m.addConstr(s <= S_max, name='SalesCap')
    m.optimize()
    return m
m = solve_problem()