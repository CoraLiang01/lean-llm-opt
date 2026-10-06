import gurobipy as gp
import pandas as pd
import numpy as np
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
fixed_df = pd.read_csv(fixed_costs_path, sep=',')
fixed_df['Service Center'] = fixed_df['Service Center'].astype(str).str.strip()
service_centers = list(fixed_df['Service Center'])
fixed_cost = dict(zip(fixed_df['Service Center'], fixed_df['Fixed Opening Cost']))
service_df = pd.read_csv(service_costs_path, sep=',')
service_df['Customer'] = service_df['Customer'].astype(str).str.strip()
customers = list(service_df['Customer'])
expected_sc_cols = service_centers
for sc in expected_sc_cols:
    if sc not in service_df.columns:
        raise ValueError(f"Service centre column '{sc}' missing in service cost file.")
service_cost = {}
for i, row in service_df.iterrows():
    cust = row['Customer']
    for sc in service_centers:
        service_cost[cust, sc] = float(row[sc])
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(service_centers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(customers, service_centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[j] * y[j] for j in service_centers)) + gp.quicksum((service_cost[i, j] * x[i, j] for i in customers for j in service_centers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in service_centers)) == 1 for i in customers), name='')
m.addConstrs((x[i, j] <= y[j] for i in customers for j in service_centers), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in customers)) <= 4 for j in service_centers), name='')
m.optimize()