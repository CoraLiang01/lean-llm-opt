import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', sep=',')
branches = demand_df['customer'].astype(str).tolist()
demand = demand_df.set_index('customer')['demand'].astype(float).to_dict()
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
fixed_costs = fixed_cost_df.set_index('Unnamed: 0')['fixed_costs'].astype(float).to_dict()
trans_cost_df = trans_cost_df.set_index('Unnamed: 0')
trans_cost_df.index = trans_cost_df.index.astype(str)
trans_cost_df.columns = trans_cost_df.columns.astype(str)
if not set(suppliers).issubset(set(trans_cost_df.index)):
    raise ValueError('Some suppliers in fixed_cost.csv are missing from transportation_costs.csv')
if not set(branches).issubset(set(trans_cost_df.columns)):
    raise ValueError('Some branches in demand.csv are missing from transportation_costs.csv')
transportation_costs = {(i, j): float(trans_cost_df.loc[i, j]) for i in suppliers for j in branches}
M = sum(demand.values())
m = gp.Model('UFLP_Superstore')
x = m.addVars(suppliers, branches, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_costs[i, j] * x[i, j] for i in suppliers for j in branches)), gp.GRB.MINIMIZE)
for j in branches:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in branches:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()