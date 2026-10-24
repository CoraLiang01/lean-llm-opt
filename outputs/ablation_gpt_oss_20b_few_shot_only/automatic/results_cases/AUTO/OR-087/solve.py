import pandas as pd
import gurobipy
from gurobipy import GRB
path_36_1 = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv'
path_36_2 = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv'
path_36_3 = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv'
df_36_1 = pd.read_csv(path_36_1, dtype=str, keep_default_na=False)
df_36_2 = pd.read_csv(path_36_2, dtype=str, keep_default_na=False)
df_36_3 = pd.read_csv(path_36_3, dtype=str, keep_default_na=False)
df_36_1.set_index('Product', inplace=True)
df_36_2.set_index('Product', inplace=True)
df_36_3.set_index('Product', inplace=True)
product_cols = df_36_1.columns.tolist()
if len(product_cols) != 80:
    raise ValueError(f'Expected 80 product columns, found {len(product_cols)}')

def to_float(val):
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Cannot convert value '{val}' to float")

def to_int(val):
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Cannot convert value '{val}' to int")
P = list(range(1, 81))
D = list(range(1, 23))
MaxDemand_p = {}
SellingPrice_p = {}
ProdCost_p = {}
ProdQuota_p = {}
ActivationCost_p = {}
MinBatch_p = {}
for (idx, col) in enumerate(product_cols, start=1):
    MaxDemand_p[idx] = to_int(df_36_1.loc['Maximum Demand (100 kg units)', col])
    SellingPrice_p[idx] = to_float(df_36_1.loc['Selling Price ($/100 kg)', col])
    ProdCost_p[idx] = to_float(df_36_1.loc['Production Cost ($/100 kg)', col])
    ProdQuota_p[idx] = to_int(df_36_1.loc['Production Quota (max per day)', col])
    ActivationCost_p[idx] = to_int(df_36_2.loc['Activation Cost ($)', col])
    MinBatch_p[idx] = to_int(df_36_3.loc['Minimum Batch Size (100 kg units)', col])
required_indices = set(P)
for (name, param) in [('MaxDemand_p', MaxDemand_p), ('SellingPrice_p', SellingPrice_p), ('ProdCost_p', ProdCost_p), ('ProdQuota_p', ProdQuota_p), ('ActivationCost_p', ActivationCost_p), ('MinBatch_p', MinBatch_p)]:
    if set(param.keys()) != required_indices:
        missing = required_indices - set(param.keys())
        raise ValueError(f'Missing entries in {name} for products: {sorted(missing)}')

def solve_problem():
    m = gurobipy.Model('ProductionPlanning')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(p, d) for p in P for d in D], vtype=GRB.INTEGER, lb=0, name='')
    y = m.addVars(P, vtype=GRB.BINARY, name='')
    profit_expr = gurobipy.quicksum(((SellingPrice_p[p] - ProdCost_p[p]) * x[p, d] for p in P for d in D))
    activation_expr = gurobipy.quicksum((ActivationCost_p[p] * y[p] for p in P))
    m.setObjective(profit_expr - activation_expr, GRB.MAXIMIZE)
    for p in P:
        for d in D:
            m.addConstr(x[p, d] <= ProdQuota_p[p] * y[p], name=f'quota_{p}_{d}')
    for p in P:
        m.addConstr(gurobipy.quicksum((x[p, d] for d in D)) <= MaxDemand_p[p], name=f'demand_{p}')
    for p in P:
        m.addConstr(gurobipy.quicksum((x[p, d] for d in D)) >= MinBatch_p[p] * y[p], name=f'minbatch_{p}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()