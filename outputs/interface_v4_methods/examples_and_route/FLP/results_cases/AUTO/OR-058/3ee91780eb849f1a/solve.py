import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = demand_df['customer'].tolist()
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = fixed_cost_df['supplier'].tolist()
fixed_costs = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['supplier'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
for c in customers:
    if c not in trans_cost_df.columns:
        raise KeyError(f"Customer '{c}' not found in transportation_costs.csv columns.")
transportation_costs = {}
for _, row in trans_cost_df.iterrows():
    supplier = str(row['supplier']).strip()
    for customer in customers:
        transportation_costs[supplier, customer] = float(row[customer])
if set(suppliers) != set(trans_cost_df['supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set(demand_df['customer']):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
M = sum((demand[c] for c in customers))
m = gp.Model('UFLP_Adidas_Supplier_Selection')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_term = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
transport_cost_term = gp.quicksum((transportation_costs[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()