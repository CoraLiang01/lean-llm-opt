import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = demand_df['customer'].tolist()
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['warehouse'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
warehouses = fixed_cost_df['warehouse'].tolist()
fixed_costs = dict(zip(fixed_cost_df['warehouse'], fixed_cost_df['fixed_costs']))
transport_df = pd.read_csv(transport_cost_path, sep=',')
transport_df['warehouse'] = transport_df['Unnamed: 0'].astype(str).str.strip()
transport_df = transport_df.set_index('warehouse')
transport_costs = transport_df[customers].astype(float)
if set(warehouses) != set(transport_costs.index):
    raise ValueError('Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set(transport_costs.columns):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
m = gp.Model('UFLP_Bandcamp')
x = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in warehouses)) + gp.quicksum((transport_costs.loc[i, j] * x[i, j] for i in warehouses for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouses)) == demand[j], name=f'demand_{j}')
for i in warehouses:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
m.optimize()