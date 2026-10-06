import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cost_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    fixed_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
    fixed_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
    for enc in cost_encodings:
        try:
            cost_df = pd.read_csv(cost_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Cannot read {cost_path} with tried encodings')
    for enc in fixed_encodings:
        try:
            fixed_df = pd.read_csv(fixed_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Cannot read {fixed_path} with tried encodings')
    if 'Customer' not in cost_df.columns:
        raise ValueError('Missing "Customer" column in expanded_customer_service_costs.csv')
    service_centers = [col for col in cost_df.columns if col != 'Customer']
    customers = cost_df['Customer'].tolist()
    if 'Service Center' not in fixed_df.columns or 'Fixed Opening Cost' not in fixed_df.columns:
        raise ValueError('Missing required columns in service_centers_fixed_costs.csv')
    fixed_df['Service Center'] = fixed_df['Service Center'].astype(str)
    fixed_cost_dict = {}
    for (_, row) in fixed_df.iterrows():
        sc = row['Service Center']
        if sc in fixed_cost_dict:
            raise ValueError(f'Duplicate fixed cost entry for {sc}')
        fixed_cost_dict[sc] = float(row['Fixed Opening Cost'])
    if set(service_centers) - set(fixed_cost_dict.keys()):
        missing = set(service_centers) - set(fixed_cost_dict.keys())
        raise ValueError(f'Missing fixed cost for service centers: {missing}')
    cost = {i: {} for i in service_centers}
    for (_, row) in cost_df.iterrows():
        cust = row['Customer']
        if cust in cost['SC1']:
            raise ValueError(f'Duplicate customer row for {cust}')
        for i in service_centers:
            if pd.isnull(row[i]):
                raise ValueError(f'Missing service cost for {i}, {cust}')
            cost[i][cust] = float(row[i])
    if set(customers) != set(cost_df['Customer']):
        raise ValueError('Mismatch in customer identifiers')
    for i in service_centers:
        if set(cost[i].keys()) != set(customers):
            raise ValueError(f'Missing cost entries for {i}')
    m = gp.Model('UFLP13')
    m.Params.MIPGap = 0.0001
    y = m.addVars(service_centers, vtype=GRB.BINARY, name='')
    x = m.addVars(service_centers, customers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost_dict[i] * y[i] for i in service_centers)) + gp.quicksum((cost[i][j] * x[i, j] for i in service_centers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in service_centers)) == 1 for j in customers), name='')
    m.addConstrs((x[i, j] <= y[i] for i in service_centers for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= 4 * y[i] for i in service_centers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()