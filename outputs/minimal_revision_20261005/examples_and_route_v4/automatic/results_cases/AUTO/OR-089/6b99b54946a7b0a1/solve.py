import gurobipy as gp
import pandas as pd
import numpy as np
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
df_service = pd.read_csv(service_costs_path, sep=',')
df_fixed = pd.read_csv(fixed_costs_path, sep=',')
df_service['Customer'] = df_service['Customer'].astype(str).str.strip()
df_fixed['Service Center'] = df_fixed['Service Center'].astype(str).str.strip()
customers = [f'C{i}' for i in range(1, 16)]
centers = [f'SC{i}' for i in range(1, 11)]
missing_customers = set(customers) - set(df_service['Customer'])
if missing_customers:
    raise ValueError(f'Missing customers in service cost data: {missing_customers}')
missing_centers = set(centers) - set(df_fixed['Service Center'])
if missing_centers:
    raise ValueError(f'Missing centers in fixed cost data: {missing_centers}')
for sc in centers:
    if sc not in df_service.columns:
        raise ValueError(f'Missing service cost column for center {sc} in service cost data.')
fixed_cost = {}
for (_, row) in df_fixed.iterrows():
    sc = str(row['Service Center']).strip()
    if sc in centers:
        fixed_cost[sc] = float(row['Fixed Opening Cost'])
if set(fixed_cost.keys()) != set(centers):
    raise ValueError('Fixed cost data does not cover all required centers.')
service_cost = {}
for (_, row) in df_service.iterrows():
    cust = str(row['Customer']).strip()
    if cust in customers:
        for sc in centers:
            service_cost[cust, sc] = float(row[sc])
if set(((c, sc) for c in customers for sc in centers)) != set(service_cost.keys()):
    raise ValueError('Service cost data does not cover all required customer-center pairs.')
m = gp.Model('UFLP_Capacitated')
m.Params.MIPGap = 0.0001
y = m.addVars(centers, vtype=gp.GRB.BINARY, name='')
x = m.addVars([(i, j) for i in customers for j in centers], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[j] * y[j] for j in centers)) + gp.quicksum((service_cost[i, j] * x[i, j] for i in customers for j in centers)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in centers)) == 1 for i in customers), name='')
m.addConstrs((x[i, j] <= y[j] for i in customers for j in centers), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in customers)) <= 4 for j in centers), name='')
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')