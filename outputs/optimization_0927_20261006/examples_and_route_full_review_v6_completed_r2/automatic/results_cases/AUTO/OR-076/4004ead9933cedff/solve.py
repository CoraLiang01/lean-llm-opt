import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
warehouse_df = pd.read_csv(warehouse_path, dtype=str, keep_default_na=False)
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
warehouse_ids = warehouse_df['Warehouse ID'].astype(str).tolist()
customer_ids = demand_df['Customer ID'].astype(str).tolist()
fixed_cost = {}
capacity = {}
for (idx, row) in warehouse_df.iterrows():
    wid = str(row['Warehouse ID'])
    try:
        fixed_cost[wid] = float(row['Fixed_Cost'])
        capacity[wid] = float(row['Capacity'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in warehouse.csv for warehouse {wid}: {e}')
demand = {}
for (idx, row) in demand_df.iterrows():
    cid = str(row['Customer ID'])
    try:
        demand[cid] = float(row['Demand'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in demand.csv for customer {cid}: {e}')
cost = {}
cost_df_indexed = cost_df.set_index('Warehouse ID')
for wid in warehouse_ids:
    if wid not in cost_df_indexed.index:
        raise KeyError(f'Warehouse ID {wid} from warehouse.csv not found in cost.csv')
    row = cost_df_indexed.loc[wid]
    cost[wid] = {}
    for cid in customer_ids:
        if cid not in cost_df.columns:
            raise KeyError(f'Customer ID {cid} from demand.csv not found as column in cost.csv')
        try:
            cost[wid][cid] = float(row[cid])
        except Exception as e:
            raise ValueError(f'Invalid numeric value in cost.csv for warehouse {wid}, customer {cid}: {e}')
if set(warehouse_ids) != set(cost_df_indexed.index):
    raise ValueError('Mismatch between warehouse IDs in warehouse.csv and cost.csv')
if not all((cid in cost_df.columns for cid in customer_ids)):
    raise ValueError('Mismatch between customer IDs in demand.csv and columns in cost.csv')
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(warehouse_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[wid] * y_vars[wid] for wid in warehouse_ids)) + gp.quicksum((cost[wid][cid] * x_vars[wid, cid] for wid in warehouse_ids for cid in customer_ids)), gp.GRB.MINIMIZE)
for cid in customer_ids:
    m.addConstr(gp.quicksum((x_vars[wid, cid] for wid in warehouse_ids)) == demand[cid])
for wid in warehouse_ids:
    m.addConstr(gp.quicksum((x_vars[wid, cid] for cid in customer_ids)) <= capacity[wid] * y_vars[wid])
m.optimize()