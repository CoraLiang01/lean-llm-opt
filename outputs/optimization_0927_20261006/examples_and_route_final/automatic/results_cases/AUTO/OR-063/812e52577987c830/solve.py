import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
transport_cost_df = pd.read_csv(transport_cost_path, dtype=str, keep_default_na=False)
customers = demand_df['customer'].astype(str).str.strip().tolist()
warehouses = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
transport_warehouses = transport_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
transport_customers = [col for col in transport_cost_df.columns if col != 'Unnamed: 0']
if set(warehouses) != set(transport_warehouses):
    raise ValueError(f'Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv: {set(warehouses)} vs {set(transport_warehouses)}')
if set(customers) != set(transport_customers):
    raise ValueError(f'Mismatch between customers in demand.csv and transportation_costs.csv: {set(customers)} vs {set(transport_customers)}')
fixed_cost = {}
for (idx, row) in fixed_cost_df.iterrows():
    wid = str(row['Unnamed: 0']).strip()
    try:
        fixed_cost[wid] = float(row['fixed_costs'])
    except Exception as e:
        raise ValueError(f"Invalid fixed_costs value for warehouse {wid}: {row['fixed_costs']}") from e
demand = {}
for (idx, row) in demand_df.iterrows():
    cid = str(row['customer']).strip()
    try:
        demand[cid] = int(row['demand'])
    except Exception as e:
        raise ValueError(f"Invalid demand value for customer {cid}: {row['demand']}") from e
transport_cost = {}
for (idx, row) in transport_cost_df.iterrows():
    wid = str(row['Unnamed: 0']).strip()
    for cid in customers:
        try:
            transport_cost[wid, cid] = float(row[cid])
        except Exception as e:
            raise ValueError(f'Invalid transportation cost for warehouse {wid}, customer {cid}: {row[cid]}') from e
m = gp.Model('UFLP_Bandcamp')
y_vars = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouses)) + gp.quicksum((transport_cost[i, j] * x_vars[i, j] for i in warehouses for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j], name=f'demand_{j}')
for i in warehouses:
    for j in customers:
        m.addConstr(x_vars[i, j] <= demand[j] * y_vars[i], name=f'link_{i}_{j}')
m.optimize()