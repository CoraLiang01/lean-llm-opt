import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_demand['customer'] = df_demand['customer'].astype(str).str.strip()
customers = list(df_demand['customer'])
demand_dict = dict(zip(df_demand['customer'], df_demand['demand']))
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_supply['warehouse'] = df_supply['Unnamed: 0'].astype(str).str.strip()
warehouses = list(df_supply['warehouse'])
supply_dict = dict(zip(df_supply['warehouse'], df_supply['supply_capacity']))
df_cost = pd.read_csv(transportation_costs_path, sep=',')
df_cost['warehouse'] = df_cost['Unnamed: 0'].astype(str).str.strip()
cost_columns = [c for c in df_cost.columns if c in customers]
if set(customers) != set(cost_columns):
    raise ValueError(f'Mismatch between customers in demand and cost files: {set(customers)} vs {set(cost_columns)}')
if set(warehouses) != set(df_cost['warehouse']):
    raise ValueError(f"Mismatch between warehouses in supply and cost files: {set(warehouses)} vs {set(df_cost['warehouse'])}")
cost = {}
for _, row in df_cost.iterrows():
    s = str(row['warehouse']).strip()
    for c in customers:
        cost[s, c] = float(row[c])
for s in warehouses:
    if s not in supply_dict:
        raise ValueError(f'Warehouse {s} missing in supply_capacity.csv')
    for c in customers:
        if (s, c) not in cost:
            raise ValueError(f'Missing cost for warehouse {s} to customer {c}')
for c in customers:
    if c not in demand_dict:
        raise ValueError(f'Customer {c} missing in customer_demand.csv')
m = gp.Model('TransportationProblem')
x = m.addVars(warehouses, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x[s, c] for s in warehouses for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in warehouses)) == demand_dict[c], name=f'demand_{c}')
for s in warehouses:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_dict[s], name=f'supply_{s}')
m.optimize()