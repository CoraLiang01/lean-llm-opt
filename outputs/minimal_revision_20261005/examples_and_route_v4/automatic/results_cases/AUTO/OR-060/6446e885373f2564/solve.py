import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = demand_df['customer'].tolist()
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = fixed_cost_df['supplier'].tolist()
fixed_cost_dict = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['supplier'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
missing_customers = [c for c in customers if c not in trans_cost_df.columns]
if missing_customers:
    raise ValueError(f'Missing transportation cost columns for customers: {missing_customers}')
missing_suppliers = [s for s in suppliers if s not in trans_cost_df['supplier'].values]
if missing_suppliers:
    raise ValueError(f'Missing transportation cost rows for suppliers: {missing_suppliers}')
transport_cost = {}
for (_, row) in trans_cost_df.iterrows():
    supplier = str(row['supplier']).strip()
    for customer in customers:
        val = row[customer]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {supplier}, customer {customer}')
        transport_cost[supplier, customer] = float(val)
if set(suppliers) != set(trans_cost_df['supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set(demand_df['customer']):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
m = gp.Model('UFLP')
x = m.addVars([(i, j) for i in suppliers for j in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost_dict[i] * y[i] for i in suppliers)) + gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= demand_dict[j] * y[i], name=f'activate_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')