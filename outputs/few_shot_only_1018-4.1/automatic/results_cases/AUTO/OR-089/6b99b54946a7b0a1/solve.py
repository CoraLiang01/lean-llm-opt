import pandas as pd
import gurobipy as gp
from gurobipy import GRB
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
fixed_df = pd.read_csv(fixed_costs_path, dtype=str, keep_default_na=False)
fixed_df = fixed_df.rename(columns=lambda x: x.strip())
fixed_df['Service Center'] = fixed_df['Service Center'].str.strip()
fixed_df['Fixed Opening Cost'] = fixed_df['Fixed Opening Cost'].astype(float)
centers = list(fixed_df['Service Center'])
fixed_cost = dict(zip(fixed_df['Service Center'], fixed_df['Fixed Opening Cost']))
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
service_df = pd.read_csv(service_costs_path, dtype=str, keep_default_na=False, delim_whitespace=False)
service_df = service_df.rename(columns=lambda x: x.strip())
service_df['Customer'] = service_df['Customer'].str.strip()
customers = list(service_df['Customer'])
service_df = service_df.set_index('Customer')
service_df = service_df.applymap(lambda x: float(x.strip()))
cost = {i: {j: service_df.loc[j, i] for j in customers} for i in centers}
if set(cost.keys()) != set(centers):
    raise ValueError('Mismatch in centers between fixed costs and service costs.')
for i in centers:
    if set(cost[i].keys()) != set(customers):
        raise ValueError(f'Mismatch in customers for center {i}.')
m = gp.Model('UFLP13')
y_vars = m.addVars(centers, vtype=GRB.BINARY, name='')
x_vars = m.addVars(centers, customers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in centers)) + gp.quicksum((cost[i][j] * x_vars[i, j] for i in centers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in centers)) == 1 for j in customers), name='')
m.addConstrs((x_vars[i, j] <= y_vars[i] for i in centers for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= 4 * y_vars[i] for i in centers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')