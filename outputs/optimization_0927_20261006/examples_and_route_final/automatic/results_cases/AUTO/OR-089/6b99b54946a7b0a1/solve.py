import gurobipy as gp
import pandas as pd
import numpy as np
import re
fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv', sep=',', dtype=str, keep_default_na=False)
fixed_costs_df['Service Center'] = fixed_costs_df['Service Center'].str.strip()
fixed_costs_df['Fixed Opening Cost'] = fixed_costs_df['Fixed Opening Cost'].astype(float)
service_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv', sep=',', dtype=str, keep_default_na=False)
service_costs_df['Customer'] = service_costs_df['Customer'].str.strip()
sc_columns = [col for col in service_costs_df.columns if re.fullmatch('SC\\d+', col)]
for col in sc_columns:
    service_costs_df[col] = service_costs_df[col].astype(float)
service_centers = [f'SC{i}' for i in range(1, 11)]
customers = [f'C{i}' for i in range(1, 16)]
missing_scs = set(service_centers) - set(fixed_costs_df['Service Center'])
if missing_scs:
    raise ValueError(f'Missing service centers in fixed costs CSV: {missing_scs}')
missing_customers = set(customers) - set(service_costs_df['Customer'])
if missing_customers:
    raise ValueError(f'Missing customers in service costs CSV: {missing_customers}')
missing_sc_cols = set(service_centers) - set(sc_columns)
if missing_sc_cols:
    raise ValueError(f'Missing service center columns in service costs CSV: {missing_sc_cols}')
fixed_opening_cost = {row['Service Center']: row['Fixed Opening Cost'] for (_, row) in fixed_costs_df.iterrows() if row['Service Center'] in service_centers}
service_cost = {}
for (_, row) in service_costs_df.iterrows():
    c = row['Customer']
    if c in customers:
        for s in service_centers:
            service_cost[c, s] = float(row[s])
m = gp.Model('UFLP_Capacitated')
y_vars = m.addVars(service_centers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(customers, service_centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_opening_cost[s] * y_vars[s] for s in service_centers)) + gp.quicksum((service_cost[c, s] * x_vars[c, s] for c in customers for s in service_centers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[c, s] for s in service_centers)) == 1 for c in customers), name='')
m.addConstrs((x_vars[c, s] <= y_vars[s] for c in customers for s in service_centers), name='')
m.addConstrs((gp.quicksum((x_vars[c, s] for c in customers)) <= 4 for s in service_centers), name='')
m.optimize()