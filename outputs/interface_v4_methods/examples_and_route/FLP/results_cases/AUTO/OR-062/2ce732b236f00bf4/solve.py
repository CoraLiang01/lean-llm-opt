import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['Customer'] = demand_df['Customer'].astype(str).str.strip()
customers = list(demand_df['Customer'])
demand = dict(zip(demand_df['Customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['Supplier'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = list(fixed_cost_df['Supplier'])
fixed_costs = dict(zip(fixed_cost_df['Supplier'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['Supplier'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
trans_cost_df = trans_cost_df.set_index('Supplier')
trans_cost_df.columns = [str(c).strip() for c in trans_cost_df.columns]
customer_cols = [c for c in trans_cost_df.columns if c in customers]
if set(customer_cols) != set(customers):
    raise ValueError(f'Mismatch between customers in demand.csv and transportation_costs.csv columns: {set(customers)} vs {set(customer_cols)}')
if set(trans_cost_df.index) != set(suppliers):
    raise ValueError(f'Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(trans_cost_df.index)}')
transport_cost = {}
for i in suppliers:
    for j in customers:
        val = trans_cost_df.loc[i, j]
        if pd.isnull(val):
            raise ValueError(f"Missing transportation cost for supplier '{i}', customer '{j}'")
        transport_cost[i, j] = float(val)
M = sum(demand.values())
m = gp.Model('UFLP_Iowa_Liquor')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_term = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
transport_cost_term = gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()