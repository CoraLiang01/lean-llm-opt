import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].unique().tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['plant'] = df_supply['Unnamed: 0'].astype(str).str.strip()
plants = df_supply['plant'].unique().tolist()
supply_capacity = dict(zip(df_supply['plant'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['plant'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_plants = df_cost['plant'].tolist()
cost_customers = [col for col in df_cost.columns if col not in ['Unnamed: 0', 'plant']]
if set(plants) != set(cost_plants):
    raise ValueError(f'Plants in supply_capacity.csv and transportation_costs.csv do not match: {plants} vs {cost_plants}')
if set(customers) != set(cost_customers):
    raise ValueError(f'Customers in customer_demand.csv and transportation_costs.csv do not match: {customers} vs {cost_customers}')
cost = {}
for (_, row) in df_cost.iterrows():
    s = row['plant']
    for c in customers:
        val = row[c]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for plant {s}, customer {c}')
        cost[s, c] = float(val)
for s in plants:
    if s not in supply_capacity:
        raise ValueError(f'Missing supply capacity for plant {s}')
for c in customers:
    if c not in demand:
        raise ValueError(f'Missing demand for customer {c}')
for s in plants:
    for c in customers:
        if (s, c) not in cost:
            raise ValueError(f'Missing transportation cost for plant {s}, customer {c}')

def solve_problem(plants, customers, supply_capacity, demand, cost):
    m = gp.Model('BrewCo_Transportation')
    m.Params.MIPGap = 0.0001
    x = m.addVars([(s, c) for s in plants for c in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in plants for c in customers)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[s, c] for s in plants)) == demand[c] for c in customers), name='')
    m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s] for s in plants), name='')
    m.optimize()
    return m
m = solve_problem(plants, customers, supply_capacity, demand, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')