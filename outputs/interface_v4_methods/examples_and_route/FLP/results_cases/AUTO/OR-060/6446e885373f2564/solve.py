import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = list(demand_df['customer'])
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_df['supplier'] = fixed_df['Unnamed: 0'].astype(str).str.strip()
suppliers = list(fixed_df['supplier'])
fixed_cost = dict(zip(fixed_df['supplier'], fixed_df['fixed_costs']))
trans_df = pd.read_csv(transport_cost_path, sep=',')
trans_df['supplier'] = trans_df['Unnamed: 0'].astype(str).str.strip()
trans_suppliers = set(trans_df['supplier'])
if set(suppliers) != trans_suppliers:
    raise ValueError(f'Mismatch in supplier IDs between fixed_cost.csv and transportation_costs.csv: {set(suppliers) ^ trans_suppliers}')
trans_customers = [col for col in trans_df.columns if col.startswith('C')]
if set(customers) != set(trans_customers):
    raise ValueError(f'Mismatch in customer IDs between demand.csv and transportation_costs.csv: {set(customers) ^ set(trans_customers)}')
transport_cost = {}
for _, row in trans_df.iterrows():
    i = row['supplier']
    for j in customers:
        transport_cost[i, j] = float(row[j])
m = gp.Model('UFLP')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'activate_{i}_{j}')
m.optimize()