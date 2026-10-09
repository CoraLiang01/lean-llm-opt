import gurobipy as gp
import pandas as pd
import numpy as np
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
fixed_costs_df = pd.read_csv(fixed_costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Service Center' not in fixed_costs_df.columns or 'Fixed Opening Cost' not in fixed_costs_df.columns:
    raise KeyError('Missing required columns in service_centers_fixed_costs.csv')
fixed_costs_df['Service Center'] = fixed_costs_df['Service Center'].str.strip()
fixed_costs_df['Fixed Opening Cost'] = fixed_costs_df['Fixed Opening Cost'].astype(float)
service_centers = [f'SC{i}' for i in range(1, 11)]
fixed_costs_dict = {}
for sc in service_centers:
    match = fixed_costs_df[fixed_costs_df['Service Center'].str.casefold() == sc.casefold()]
    if match.empty:
        raise ValueError(f'Service center {sc} not found in fixed costs file.')
    fixed_costs_dict[sc] = float(match['Fixed Opening Cost'].iloc[0])
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
service_costs_df = pd.read_csv(service_costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Customer' not in service_costs_df.columns:
    raise KeyError("Missing 'Customer' column in expanded_customer_service_costs.csv")
service_costs_df['Customer'] = service_costs_df['Customer'].str.strip()
customers = [f'C{i}' for i in range(1, 16)]
for c in customers:
    if not (service_costs_df['Customer'].str.casefold() == c.casefold()).any():
        raise ValueError(f'Customer {c} not found in service costs file.')
for sc in service_centers:
    if sc not in service_costs_df.columns:
        raise KeyError(f'Service center column {sc} missing in expanded_customer_service_costs.csv')
service_costs_dict = {}
for c in customers:
    row = service_costs_df[service_costs_df['Customer'].str.casefold() == c.casefold()]
    if row.empty:
        raise ValueError(f'Customer {c} not found in service costs file.')
    row = row.iloc[0]
    for sc in service_centers:
        try:
            cost = float(row[sc])
        except Exception:
            raise ValueError(f'Invalid or missing service cost for customer {c}, center {sc}.')
        service_costs_dict[c, sc] = cost
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(service_centers, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(customers, service_centers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_costs_dict[sc] * y_vars[sc] for sc in service_centers)) + gp.quicksum((service_costs_dict[c, sc] * x_vars[c, sc] for c in customers for sc in service_centers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[c, sc] for sc in service_centers)) == 1 for c in customers), name='')
m.addConstrs((x_vars[c, sc] <= y_vars[sc] for c in customers for sc in service_centers), name='')
m.addConstrs((gp.quicksum((x_vars[c, sc] for c in customers)) <= 4 * y_vars[sc] for sc in service_centers), name='')
m.optimize()