import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].unique().tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['Unnamed: 0'] = df_supply['Unnamed: 0'].astype(str).str.strip()
sources = df_supply['Unnamed: 0'].unique().tolist()
supply_capacity = dict(zip(df_supply['Unnamed: 0'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_sources = df_cost['Unnamed: 0'].tolist()
cost_customers = [col for col in df_cost.columns if col != 'Unnamed: 0']
if set(sources) != set(cost_sources):
    raise ValueError(f'Mismatch between supply sources in supply_capacity.csv and transportation_costs.csv: {set(sources) ^ set(cost_sources)}')
if set(customers) != set(cost_customers):
    raise ValueError(f'Mismatch between customers in customer_demand.csv and transportation_costs.csv: {set(customers) ^ set(cost_customers)}')
cost = {}
for (_, row) in df_cost.iterrows():
    s = row['Unnamed: 0']
    for c in customers:
        val = row[c]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for source {s}, customer {c}')
        cost[s, c] = float(val)
for s in sources:
    if s not in supply_capacity:
        raise ValueError(f'Supply capacity missing for source {s}')
for c in customers:
    if c not in demand:
        raise ValueError(f'Demand missing for customer {c}')
for s in sources:
    for c in customers:
        if (s, c) not in cost:
            raise ValueError(f'Transportation cost missing for ({s}, {c})')

def solve_transportation_problem(sources, customers, supply_capacity, demand, cost):
    m = gp.Model('Transportation')
    x = m.addVars([(s, c) for s in sources for c in customers], lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in sources for c in customers)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s] for s in sources), name='')
    m.addConstrs((gp.quicksum((x[s, c] for s in sources)) == demand[c] for c in customers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_transportation_problem(sources, customers, supply_capacity, demand, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')