import gurobipy as gp
import pandas as pd
import numpy as np
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
service_costs_df = pd.read_csv(service_costs_path, sep=',', dtype=str, keep_default_na=False)
fixed_costs_df = pd.read_csv(fixed_costs_path, sep=',', dtype=str, keep_default_na=False)
customers = service_costs_df['Customer'].astype(str).str.strip().tolist()
service_centers = fixed_costs_df['Service Center'].astype(str).str.strip().tolist()
expected_sc_cols = service_centers
actual_sc_cols = [c for c in service_costs_df.columns if c != 'Customer']
if set(expected_sc_cols) != set(actual_sc_cols):
    raise ValueError(f'Service center columns in service cost file do not match those in fixed cost file.\nExpected: {expected_sc_cols}\nFound: {actual_sc_cols}')
service_cost = {}
for (_, row) in service_costs_df.iterrows():
    cust = str(row['Customer']).strip()
    for sc in service_centers:
        val = row[sc]
        try:
            cost = float(val)
        except Exception:
            raise ValueError(f'Invalid service cost for customer {cust}, center {sc}: {val}')
        service_cost[cust, sc] = cost
fixed_cost = {}
for (_, row) in fixed_costs_df.iterrows():
    sc = str(row['Service Center']).strip()
    val = row['Fixed Opening Cost']
    try:
        cost = float(val)
    except Exception:
        raise ValueError(f'Invalid fixed opening cost for center {sc}: {val}')
    fixed_cost[sc] = cost
if set(service_cost.keys()) != set(((i, j) for i in customers for j in service_centers)):
    raise ValueError('Service cost data missing for some (customer, center) pairs.')
if set(fixed_cost.keys()) != set(service_centers):
    raise ValueError('Fixed cost data missing for some service centers.')
m = gp.Model('UFLP_Capacitated')
y_vars = m.addVars(service_centers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars([(i, j) for i in customers for j in service_centers], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[j] * y_vars[j] for j in service_centers)) + gp.quicksum((service_cost[i, j] * x_vars[i, j] for i in customers for j in service_centers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in service_centers)) == 1 for i in customers), name='')
m.addConstrs((x_vars[i, j] <= y_vars[j] for i in customers for j in service_centers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in customers)) <= 4 for j in service_centers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')