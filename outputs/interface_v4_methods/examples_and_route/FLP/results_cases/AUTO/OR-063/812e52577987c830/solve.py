import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
customers = demand_df['customer'].astype(str).str.strip().tolist()
demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
warehouses = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['Unnamed: 0'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
trans_cost_columns = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
if set(customers) != set(trans_cost_columns):
    raise ValueError(f'Customer columns in transportation_costs.csv do not match demand.csv: {set(customers)} vs {set(trans_cost_columns)}')
if set(warehouses) != set(trans_cost_df['Unnamed: 0']):
    raise ValueError(f"Warehouse IDs in fixed_cost.csv and transportation_costs.csv do not match: {set(warehouses)} vs {set(trans_cost_df['Unnamed: 0'])}")
transport_costs = {}
for _, row in trans_cost_df.iterrows():
    i = row['Unnamed: 0']
    for j in customers:
        transport_costs[i, j] = float(row[j])

def solve_uflp(warehouses, customers, fixed_costs, transport_costs, demand):
    m = gp.Model('UFLP_Bandcamp')
    x = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in warehouses)) + gp.quicksum((transport_costs[i, j] * x[i, j] for i in warehouses for j in customers)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
    m.addConstrs((x[i, j] <= demand[j] * y[i] for i in warehouses for j in customers), name='')
    m.optimize()
    return m
m = solve_uflp(warehouses, customers, fixed_costs, transport_costs, demand)