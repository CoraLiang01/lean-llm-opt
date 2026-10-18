import gurobipy as gp
import pandas as pd
import numpy as np
cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv', sep=',', dtype=str, keep_default_na=False)
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv', sep=',', dtype=str, keep_default_na=False)
cost_df['plant'] = cost_df['plant'].str.strip()
demand_df['customer'] = demand_df['customer'].str.strip()
plants = [f'F{i}' for i in range(1, 16)]
customers = [f'C{i}' for i in range(1, 16)]
missing_plants = set(plants) - set(cost_df['plant'])
if missing_plants:
    raise ValueError(f'Missing plants in cost.csv: {missing_plants}')
missing_customers = set(customers) - set(demand_df['customer'])
if missing_customers:
    raise ValueError(f'Missing customers in demand.csv: {missing_customers}')
for c in customers:
    if c not in cost_df.columns:
        raise ValueError(f'Missing transport cost column for customer {c} in cost.csv')
fixed_cost = {}
capacity = {}
transport_cost = {}
for _, row in cost_df.iterrows():
    plant = row['plant']
    fixed_cost[plant] = int(row['fixed_cost'])
    capacity[plant] = int(row['capacity'])
    for customer in customers:
        transport_cost[plant, customer] = float(row[customer])
demand = {}
for _, row in demand_df.iterrows():
    customer = row['customer']
    demand[customer] = int(row['demand'])
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(plants, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(plants, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in plants)) + gp.quicksum((transport_cost[i, j] * x_vars[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()