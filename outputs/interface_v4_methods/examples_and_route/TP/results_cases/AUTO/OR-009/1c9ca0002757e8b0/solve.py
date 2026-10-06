import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = df_demand['customer'].tolist()
customer_demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['plant'] = df_supply['Unnamed: 0'].astype(str).str.strip()
plants = df_supply['plant'].tolist()
supply_capacity = dict(zip(df_supply['plant'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['plant'] = df_cost['Unnamed: 0'].astype(str).str.strip()
for c in customers:
    if c not in df_cost.columns:
        raise KeyError(f"Customer '{c}' not found in transportation_costs.csv columns.")
cost = {}
for _, row in df_cost.iterrows():
    plant = row['plant']
    for c in customers:
        cost[plant, c] = float(row[c])
for s in plants:
    for c in customers:
        if (s, c) not in cost:
            raise KeyError(f"Missing transportation cost for plant '{s}' to customer '{c}'.")
for c in customers:
    if c not in customer_demand:
        raise KeyError(f"Missing demand for customer '{c}'.")
for s in plants:
    if s not in supply_capacity:
        raise KeyError(f"Missing supply capacity for plant '{s}'.")

def solve_brewco_transportation(plants, customers, cost, supply_capacity, customer_demand):
    m = gp.Model('BrewCo_Transportation')
    x = m.addVars(plants, customers, lb=0.0, name='')
    m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in plants for c in customers)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[s, c] for s in plants)) == customer_demand[c] for c in customers), name='')
    m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s] for s in plants), name='')
    m.optimize()
    return m
m = solve_brewco_transportation(plants, customers, cost, supply_capacity, customer_demand)