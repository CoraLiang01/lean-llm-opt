import gurobipy as gp
import pandas as pd
import numpy as np
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
service_costs_df = pd.read_csv(service_costs_path, dtype=str, keep_default_na=False)
fixed_costs_df = pd.read_csv(fixed_costs_path, dtype=str, keep_default_na=False)
customers = service_costs_df['Customer'].str.strip().tolist()
service_centers = fixed_costs_df['Service Center'].str.strip().tolist()
service_costs_columns = [col.strip() for col in service_costs_df.columns if col.strip() != 'Customer']
if set(service_centers) != set(service_costs_columns):
    raise ValueError(f'Mismatch between service centers in fixed costs and service cost columns: {set(service_centers)} vs {set(service_costs_columns)}')
fixed_costs_df['Fixed Opening Cost'] = fixed_costs_df['Fixed Opening Cost'].astype(float)
fixed_opening_cost = dict(zip(service_centers, fixed_costs_df['Fixed Opening Cost']))
for sc in service_centers:
    service_costs_df[sc] = service_costs_df[sc].astype(float)
service_costs = {}
for (_, row) in service_costs_df.iterrows():
    c = row['Customer'].strip()
    for s in service_centers:
        service_costs[c, s] = float(row[s])
m = gp.Model('UFLP_Capacitated')
y_vars = m.addVars(service_centers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(customers, service_centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_opening_cost[s] * y_vars[s] for s in service_centers)) + gp.quicksum((service_costs[c, s] * x_vars[c, s] for c in customers for s in service_centers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[c, s] for s in service_centers)) == 1, name=f'assign_{c}')
for c in customers:
    for s in service_centers:
        m.addConstr(x_vars[c, s] <= y_vars[s], name=f'open_assign_{c}_{s}')
for s in service_centers:
    m.addConstr(gp.quicksum((x_vars[c, s] for c in customers)) <= 4, name=f'capacity_{s}')
m.optimize()