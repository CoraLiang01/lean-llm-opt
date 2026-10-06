import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
warehouse_df = pd.read_csv(warehouse_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
warehouse_ids_warehouse = set(warehouse_df['Warehouse ID'].astype(str))
warehouse_ids_cost = set(cost_df['Warehouse ID'].astype(str))
warehouse_ids = sorted(list(warehouse_ids_warehouse & warehouse_ids_cost))
if len(warehouse_ids) == 0:
    raise ValueError('No matching warehouse IDs between warehouse.csv and cost.csv.')
customer_ids_demand = set(demand_df['Customer ID'].astype(str))
cost_customer_cols = [col for col in cost_df.columns if re.fullmatch('C\\d+', col)]
customer_ids_cost = set(cost_customer_cols)
customer_ids = sorted(list(customer_ids_demand & set([c for c in cost_customer_cols])))
if len(customer_ids) == 0:
    raise ValueError('No matching customer IDs between demand.csv and cost.csv columns.')
fixed_cost = {}
capacity = {}
warehouse_df_idx = warehouse_df.set_index('Warehouse ID')
for wid in warehouse_ids:
    if wid not in warehouse_df_idx.index:
        raise ValueError(f'Warehouse ID {wid} missing in warehouse.csv')
    fixed_cost[wid] = float(warehouse_df_idx.loc[wid, 'Fixed_Cost'])
    capacity[wid] = float(warehouse_df_idx.loc[wid, 'Capacity'])
demand = {}
demand_df_idx = demand_df.set_index('Customer ID')
for cid in customer_ids:
    if cid not in demand_df_idx.index:
        raise ValueError(f'Customer ID {cid} missing in demand.csv')
    demand[cid] = float(demand_df_idx.loc[cid, 'Demand'])
cost = {}
cost_df_idx = cost_df.set_index('Warehouse ID')
for wid in warehouse_ids:
    if wid not in cost_df_idx.index:
        raise ValueError(f'Warehouse ID {wid} missing in cost.csv')
    cost[wid] = {}
    for cid in customer_ids:
        if cid not in cost_customer_cols:
            raise ValueError(f'Customer column {cid} missing in cost.csv')
        val = cost_df_idx.loc[wid, cid]
        if pd.isnull(val):
            raise ValueError(f'Missing cost for warehouse {wid}, customer {cid}')
        cost[wid][cid] = float(val)
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouse_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in warehouse_ids)) + gp.quicksum((cost[i][j] * x[i, j] for i in warehouse_ids for j in customer_ids)), gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x[i, j] for j in customer_ids)) <= capacity[i], name=f'capacity_{i}')
for i in warehouse_ids:
    for j in customer_ids:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Warehouses Opened ---')
    for i in warehouse_ids:
        if y[i].X > 0.5:
            print(f'  Warehouse {i}: OPEN (Fixed cost: {fixed_cost[i]:.2f}, Capacity: {capacity[i]:.0f})')
    print('\n--- Customer Assignments ---')
    for j in customer_ids:
        print(f'Customer {j} (Demand: {demand[j]:.0f}):')
        for i in warehouse_ids:
            if x[i, j].X > 1e-06:
                print(f'  Served {x[i, j].X:.2f} from Warehouse {i} (Unit cost: {cost[i][j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')