import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
warehouse_df = pd.read_csv(warehouse_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
warehouse_ids = warehouse_df['Warehouse ID'].astype(str).tolist()
customer_ids = demand_df['Customer ID'].astype(str).tolist()
cost_warehouse_ids = cost_df['Warehouse ID'].astype(str).tolist()
cost_customer_cols = [col for col in cost_df.columns if col.startswith('C')]
if set(warehouse_ids) != set(cost_warehouse_ids):
    raise ValueError('Mismatch between warehouse IDs in warehouse.csv and cost.csv')
if set(customer_ids) != set(cost_customer_cols):
    raise ValueError('Mismatch between customer IDs in demand.csv and cost.csv columns')
fixed_cost = warehouse_df.set_index('Warehouse ID')['Fixed_Cost'].astype(float).to_dict()
capacity = warehouse_df.set_index('Warehouse ID')['Capacity'].astype(float).to_dict()
demand = demand_df.set_index('Customer ID')['Demand'].astype(float).to_dict()
cost = {}
for _, row in cost_df.iterrows():
    w = str(row['Warehouse ID'])
    cost[w] = {}
    for c in customer_ids:
        cost[w][c] = float(row[c])
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouse_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[w] * y[w] for w in warehouse_ids)) + gp.quicksum((cost[w][c] * x[w, c] for w in warehouse_ids for c in customer_ids)), gp.GRB.MINIMIZE)
for c in customer_ids:
    m.addConstr(gp.quicksum((x[w, c] for w in warehouse_ids)) == demand[c], name=f'demand_{c}')
for w in warehouse_ids:
    m.addConstr(gp.quicksum((x[w, c] for c in customer_ids)) <= capacity[w], name=f'capacity_{w}')
for w in warehouse_ids:
    for c in customer_ids:
        m.addConstr(x[w, c] <= demand[c] * y[w], name=f'link_{w}_{c}')
m.optimize()