import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', sep=',', dtype=str, keep_default_na=False)
demand_df['customer'] = demand_df['customer'].str.strip()
demand_df['demand'] = demand_df['demand'].astype(int)
branches = list(demand_df['customer'])
branch_demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', sep=',', dtype=str, keep_default_na=False)
fixed_cost_df['Unnamed: 0'] = fixed_cost_df['Unnamed: 0'].str.strip()
fixed_cost_df['fixed_costs'] = fixed_cost_df['fixed_costs'].astype(float)
suppliers = list(fixed_cost_df['Unnamed: 0'])
supplier_fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', sep=',', dtype=str, keep_default_na=False)
trans_costs_df['Unnamed: 0'] = trans_costs_df['Unnamed: 0'].str.strip()
for b in branches:
    if b not in trans_costs_df.columns:
        raise KeyError(f"Branch '{b}' not found in transportation_costs.csv columns.")
transportation_cost = {}
for (idx, row) in trans_costs_df.iterrows():
    supplier = row['Unnamed: 0']
    for branch in branches:
        try:
            transportation_cost[supplier, branch] = float(row[branch])
        except Exception as e:
            raise ValueError(f"Invalid transportation cost for supplier '{supplier}', branch '{branch}': {row[branch]}") from e
m = gp.Model('UFLP_Superstore')
x_vars = m.addVars(suppliers, branches, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_term = gp.quicksum((supplier_fixed_cost[i] * y_vars[i] for i in suppliers))
transport_cost_term = gp.quicksum((transportation_cost[i, j] * x_vars[i, j] for i in suppliers for j in branches))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in branches:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == branch_demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in branches:
        m.addConstr(x_vars[i, j] <= branch_demand[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()