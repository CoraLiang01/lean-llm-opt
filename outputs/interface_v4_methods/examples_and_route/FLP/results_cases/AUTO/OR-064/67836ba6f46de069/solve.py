import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
customers = demand_df['customer'].astype(str).tolist()
demand = demand_df.set_index('customer')['demand'].to_dict()
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
fixed_costs = fixed_cost_df.set_index('Unnamed: 0')['fixed_costs'].to_dict()
transport_df = pd.read_csv(transport_cost_path, sep=',')
transport_df['Unnamed: 0'] = transport_df['Unnamed: 0'].astype(str)
transport_df = transport_df.set_index('Unnamed: 0')
if set(suppliers) != set(transport_df.index):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) - set(transport_df.columns):
    raise ValueError('Some customers in demand.csv are missing from transportation_costs.csv columns')
transport_costs = {}
for i in suppliers:
    for j in customers:
        val = transport_df.at[i, j]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
        transport_costs[i, j] = float(val)
M = sum(demand.values())
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_term = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
transport_cost_term = gp.quicksum((transport_costs[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()