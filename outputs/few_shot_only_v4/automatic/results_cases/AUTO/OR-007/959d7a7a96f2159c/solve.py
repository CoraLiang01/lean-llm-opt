import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if demand_df.shape[1] != 2:
    raise ValueError('customer_demand.csv must have exactly two columns (store, demand)')
stores = demand_df.iloc[:, 0].tolist()
demand = {}
for idx, row in demand_df.iterrows():
    store = row.iloc[0]
    try:
        val = float(row.iloc[1])
    except Exception:
        raise ValueError(f'Invalid demand value for store {store}: {row.iloc[1]}')
    demand[store] = val
supply_df = pd.read_csv(supply_path, dtype=str, keep_default_na=False)
if supply_df.shape[1] != 2:
    raise ValueError('supply_capacity.csv must have exactly two columns (warehouse, capacity)')
warehouses = supply_df.iloc[:, 0].tolist()
supply_capacity = {}
for idx, row in supply_df.iterrows():
    wh = row.iloc[0]
    try:
        val = float(row.iloc[1])
    except Exception:
        raise ValueError(f'Invalid supply value for warehouse {wh}: {row.iloc[1]}')
    supply_capacity[wh] = val
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
if cost_df.shape[0] != len(warehouses):
    raise ValueError('transportation_costs.csv row count does not match number of warehouses')
if cost_df.shape[1] != len(stores) + 1:
    raise ValueError('transportation_costs.csv column count does not match number of stores + 1 (for warehouse id)')
if list(cost_df.columns)[1:] != stores:
    raise ValueError('transportation_costs.csv columns (excluding first) do not match store identifiers')
cost = {}
for i, row in cost_df.iterrows():
    wh = row.iloc[0]
    if wh not in warehouses:
        raise ValueError(f'Warehouse {wh} in transportation_costs.csv not found in supply_capacity.csv')
    cost[wh] = {}
    for j, store in enumerate(stores):
        try:
            val = float(row.iloc[j + 1])
        except Exception:
            raise ValueError(f'Invalid cost value for warehouse {wh}, store {store}: {row.iloc[j + 1]}')
        cost[wh][store] = val
if set(demand.keys()) != set(stores):
    raise ValueError('Mismatch between store identifiers in demand and cost matrix')
if set(supply_capacity.keys()) != set(warehouses):
    raise ValueError('Mismatch between warehouse identifiers in supply and cost matrix')
for wh in warehouses:
    if set(cost[wh].keys()) != set(stores):
        raise ValueError(f'Cost matrix for warehouse {wh} does not cover all stores')
m = gp.Model('GreenMart_Transportation')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[wh][store] * x_vars[wh, store] for wh in warehouses for store in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[wh, store] for wh in warehouses)) >= demand[store] for store in stores), name='')
m.addConstrs((gp.quicksum((x_vars[wh, store] for store in stores)) <= supply_capacity[wh] for wh in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')