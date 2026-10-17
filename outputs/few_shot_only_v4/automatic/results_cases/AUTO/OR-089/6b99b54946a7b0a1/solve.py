import gurobipy as gp
from gurobipy import GRB
import pandas as pd
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
fixed_costs_df = pd.read_csv(fixed_costs_path, dtype=str, keep_default_na=False)
if 'ServiceCenter' not in fixed_costs_df.columns or 'FixedCost' not in fixed_costs_df.columns:
    raise ValueError('service_centers_fixed_costs.csv must have columns ServiceCenter and FixedCost')
service_centers = fixed_costs_df['ServiceCenter'].tolist()
fixed_cost = {}
for idx, row in fixed_costs_df.iterrows():
    try:
        fixed_cost[row['ServiceCenter']] = float(row['FixedCost'])
    except Exception:
        raise ValueError(f"Invalid FixedCost for {row['ServiceCenter']}")
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
service_costs_df = pd.read_csv(service_costs_path, dtype=str, keep_default_na=False)
if 'Customer' not in service_costs_df.columns:
    raise ValueError('expanded_customer_service_costs.csv must have a Customer column')
customers = service_costs_df['Customer'].tolist()
for sc in service_centers:
    if sc not in service_costs_df.columns:
        raise ValueError(f'Service center {sc} missing in expanded_customer_service_costs.csv')
cost = {}
for i, row in service_costs_df.iterrows():
    cust = row['Customer']
    cost[cust] = {}
    for sc in service_centers:
        try:
            cost[cust][sc] = float(row[sc])
        except Exception:
            raise ValueError(f'Invalid service cost for customer {cust}, center {sc}')
cost_sc_cust = {sc: {cust: cost[cust][sc] for cust in customers} for sc in service_centers}
if set(fixed_cost.keys()) != set(service_centers):
    raise ValueError('Mismatch in service center identifiers between fixed cost and service cost files')
for cust in customers:
    if set(cost[cust].keys()) != set(service_centers):
        raise ValueError(f'Missing service center costs for customer {cust}')
m = gp.Model('UFLP13')
y_vars = m.addVars(service_centers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(service_centers, customers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[sc] * y_vars[sc] for sc in service_centers)) + gp.quicksum((cost_sc_cust[sc][cust] * x_vars[sc, cust] for sc in service_centers for cust in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[sc, cust] for sc in service_centers)) == 1 for cust in customers), name='')
m.addConstrs((x_vars[sc, cust] <= y_vars[sc] for sc in service_centers for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[sc, cust] for cust in customers)) <= 4 * y_vars[sc] for sc in service_centers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')