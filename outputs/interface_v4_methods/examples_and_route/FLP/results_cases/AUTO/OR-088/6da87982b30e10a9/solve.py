import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
cost_df['plant'] = cost_df['plant'].astype(str).str.strip()
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
plants = list(cost_df['plant'])
customers = list(demand_df['customer'])
if len(plants) != 15 or len(customers) != 15:
    raise ValueError('Expected 15 plants and 15 customers as per the query.')
fixed_cost = dict(zip(cost_df['plant'], cost_df['fixed_cost']))
capacity = dict(zip(cost_df['plant'], cost_df['capacity']))
transport_cost = {}
for i, row in cost_df.iterrows():
    plant = row['plant']
    for customer in customers:
        if customer not in cost_df.columns:
            raise KeyError(f"Customer '{customer}' not found as a column in cost.csv.")
        transport_cost[plant, customer] = float(row[customer])
demand = dict(zip(demand_df['customer'], demand_df['demand']))
m = gp.Model('CapacitatedFacilityLocation')
x = m.addVars(plants, customers, lb=0.0, name='')
y = m.addVars(plants, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((transport_cost[i, j] * x[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in plants)) == demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()