import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if demand_df.shape[1] == 2:
    customer_ids = demand_df.iloc[:, 0].tolist()
    demand = {row[0]: float(row[1]) for row in demand_df.values}
else:
    raise ValueError('customer_demand.csv must have two columns: customer_id, demand')
supply_df = pd.read_csv(supply_path, dtype=str, keep_default_na=False)
if supply_df.shape[1] == 2:
    store_ids = supply_df.iloc[:, 0].tolist()
    supply_capacity = {row[0]: float(row[1]) for row in supply_df.values}
else:
    raise ValueError('supply_capacity.csv must have two columns: store_id, capacity')
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False, index_col=0)
cost_df = cost_df.apply(pd.to_numeric)
cost_store_ids = cost_df.index.tolist()
cost_customer_ids = cost_df.columns.tolist()
if set(store_ids) != set(cost_store_ids):
    raise ValueError(f'Store IDs in supply_capacity.csv and transportation_costs.csv do not match: {set(store_ids)} vs {set(cost_store_ids)}')
if set(customer_ids) != set(cost_customer_ids):
    raise ValueError(f'Customer IDs in customer_demand.csv and transportation_costs.csv do not match: {set(customer_ids)} vs {set(cost_customer_ids)}')
stores = store_ids
customers = customer_ids
cost = {i: {j: float(cost_df.loc[i, j]) for j in customers} for i in stores}
m = gp.Model('Original_RAG_TP')
x_vars = m.addVars(stores, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in stores for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in stores)) >= demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i] for i in stores), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')