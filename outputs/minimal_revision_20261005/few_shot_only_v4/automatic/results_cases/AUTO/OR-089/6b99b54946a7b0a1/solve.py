import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
    try:
        df_fixed = pd.read_csv(fixed_costs_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df_fixed = pd.read_csv(fixed_costs_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_fixed = pd.read_csv(fixed_costs_path, encoding='gbk')
            except UnicodeDecodeError:
                df_fixed = pd.read_csv(fixed_costs_path, encoding='latin-1')
    if 'Service Center' not in df_fixed.columns or 'Fixed Opening Cost' not in df_fixed.columns:
        raise ValueError('Missing required columns in service_centers_fixed_costs.csv')
    service_centers = df_fixed['Service Center'].astype(str).tolist()
    fixed_cost = dict(zip(df_fixed['Service Center'].astype(str), df_fixed['Fixed Opening Cost']))
    costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
    try:
        df_costs = pd.read_csv(costs_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df_costs = pd.read_csv(costs_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_costs = pd.read_csv(costs_path, encoding='gbk')
            except UnicodeDecodeError:
                df_costs = pd.read_csv(costs_path, encoding='latin-1')
    if 'Customer' not in df_costs.columns:
        raise ValueError("Missing 'Customer' column in expanded_customer_service_costs.csv")
    customers = df_costs['Customer'].astype(str).tolist()
    for sc in service_centers:
        if sc not in df_costs.columns:
            raise ValueError(f'Service center {sc} missing in expanded_customer_service_costs.csv columns')
    cost = {sc: {} for sc in service_centers}
    for (idx, row) in df_costs.iterrows():
        cust = str(row['Customer'])
        for sc in service_centers:
            cost[sc][cust] = row[sc]
    if set(service_centers) != set(df_costs.columns[1:]):
        raise ValueError('Mismatch between service center columns in cost file and fixed cost file')
    if set(customers) != set(df_costs['Customer'].astype(str)):
        raise ValueError('Mismatch in customer identifiers')
    m = gp.Model('UFLP13')
    y = m.addVars(service_centers, vtype=GRB.BINARY, name='')
    x = m.addVars(service_centers, customers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in service_centers)) + gp.quicksum((cost[i][j] * x[i, j] for i in service_centers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in service_centers)) == 1 for j in customers), name='')
    m.addConstrs((x[i, j] <= y[i] for i in service_centers for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= 4 * y[i] for i in service_centers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')