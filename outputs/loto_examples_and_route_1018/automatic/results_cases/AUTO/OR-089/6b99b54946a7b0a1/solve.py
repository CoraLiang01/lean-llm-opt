import gurobipy as gp
import pandas as pd
import numpy as np
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
fixed_costs_df = pd.read_csv(fixed_costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Service Center' not in fixed_costs_df.columns or 'Fixed Opening Cost' not in fixed_costs_df.columns:
    raise KeyError('Missing required columns in service_centers_fixed_costs.csv')
fixed_costs_df['Service Center'] = fixed_costs_df['Service Center'].str.strip()
fixed_costs_df['Fixed Opening Cost'] = fixed_costs_df['Fixed Opening Cost'].astype(float)
service_centers = fixed_costs_df['Service Center'].tolist()
fixed_costs = dict(zip(fixed_costs_df['Service Center'], fixed_costs_df['Fixed Opening Cost']))
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
service_costs_df = pd.read_csv(service_costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Customer' not in service_costs_df.columns:
    raise KeyError("Missing 'Customer' column in expanded_customer_service_costs.csv")
service_costs_df['Customer'] = service_costs_df['Customer'].str.strip()
customers = service_costs_df['Customer'].tolist()
for sc in service_centers:
    if sc not in service_costs_df.columns:
        raise KeyError(f"Service center column '{sc}' missing in expanded_customer_service_costs.csv")
service_costs = {}
for _, row in service_costs_df.iterrows():
    c = row['Customer']
    for s in service_centers:
        try:
            cost = float(row[s])
        except Exception as e:
            raise ValueError(f"Invalid service cost for customer '{c}', center '{s}': {row[s]}")
        service_costs[c, s] = cost
if len(service_centers) != 10:
    raise ValueError(f'Expected 10 service centers, got {len(service_centers)}')
if len(customers) != 15:
    raise ValueError(f'Expected 15 customers, got {len(customers)}')
if set(fixed_costs.keys()) != set(service_centers):
    raise ValueError('Mismatch between service center IDs in fixed costs and service cost matrix')
for c in customers:
    for s in service_centers:
        if (c, s) not in service_costs:
            raise ValueError(f"Missing service cost for customer '{c}', center '{s}'")
m = gp.Model('UFLP_Capacitated')
y_vars = m.addVars(service_centers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(customers, service_centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_costs[s] * y_vars[s] for s in service_centers)) + gp.quicksum((service_costs[c, s] * x_vars[c, s] for c in customers for s in service_centers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[c, s] for s in service_centers)) == 1 for c in customers), name='')
m.addConstrs((x_vars[c, s] <= y_vars[s] for c in customers for s in service_centers), name='')
m.addConstrs((gp.quicksum((x_vars[c, s] for c in customers)) <= 4 for s in service_centers), name='')
m.optimize()