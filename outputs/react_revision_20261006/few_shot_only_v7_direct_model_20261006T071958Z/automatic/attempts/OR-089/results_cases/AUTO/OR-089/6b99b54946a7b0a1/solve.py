import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    sc_fc_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
    try_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in try_encodings:
        try:
            sc_fc_df = pd.read_csv(sc_fc_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {sc_fc_path} with tried encodings.')
    cs_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
    for enc in try_encodings:
        try:
            cs_cost_df = pd.read_csv(cs_cost_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {cs_cost_path} with tried encodings.')
    if 'Service Center' not in sc_fc_df.columns or 'Fixed Opening Cost' not in sc_fc_df.columns:
        raise ValueError('Missing required columns in service_centers_fixed_costs.csv')
    S = list(sc_fc_df['Service Center'])
    f_s = {}
    for (_, row) in sc_fc_df.iterrows():
        sc = row['Service Center']
        try:
            fcost = float(row['Fixed Opening Cost'])
        except Exception:
            raise ValueError(f'Invalid fixed cost for service center {sc}')
        f_s[sc] = fcost
    if 'Customer' not in cs_cost_df.columns:
        raise ValueError("Missing 'Customer' column in expanded_customer_service_costs.csv")
    C = list(cs_cost_df['Customer'])
    sc_columns = [col for col in cs_cost_df.columns if col != 'Customer']
    missing_sc = set(S) - set(sc_columns)
    if missing_sc:
        raise ValueError(f'Service centers {missing_sc} missing in expanded_customer_service_costs.csv columns')
    c_sc = {}
    for (_, row) in cs_cost_df.iterrows():
        cust = row['Customer']
        c_sc[cust] = {}
        for sc in S:
            try:
                cost = float(row[sc])
            except Exception:
                raise ValueError(f'Invalid service cost for customer {cust}, service center {sc}')
            c_sc[cust][sc] = cost
    if len(S) == 0 or len(C) == 0:
        raise ValueError('No service centers or customers found in input data.')
    for cust in C:
        if set(c_sc[cust].keys()) != set(S):
            raise ValueError(f'Customer {cust} does not have costs for all service centers.')
    m = gp.Model('UFLP13')
    m.Params.MIPGap = 0.0001
    y_vars = m.addVars(S, vtype=GRB.BINARY, name='')
    x_vars = m.addVars(S, C, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f_s[s] * y_vars[s] for s in S)) + gp.quicksum((c_sc[c][s] * x_vars[s, c] for s in S for c in C)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[s, c] for s in S)) == 1 for c in C), name='')
    m.addConstrs((x_vars[s, c] <= y_vars[s] for s in S for c in C), name='')
    m.addConstrs((gp.quicksum((x_vars[s, c] for c in C)) <= 4 * y_vars[s] for s in S), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()