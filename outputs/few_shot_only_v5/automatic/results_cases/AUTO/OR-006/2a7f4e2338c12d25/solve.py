import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv', dtype=str, keep_default_na=False, skipinitialspace=True)
demand = {}
for idx, row in demand_df.iterrows():
    cust = row[demand_df.columns[0]].strip()
    val = float(row[demand_df.columns[1]])
    demand[cust] = val
supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv', dtype=str, keep_default_na=False, skipinitialspace=True)
supply_capacity = {}
for idx, row in supply_df.iterrows():
    wh = row[supply_df.columns[0]].strip()
    val = float(row[supply_df.columns[1]])
    supply_capacity[wh] = val
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv', dtype=str, keep_default_na=False, skipinitialspace=True)
warehouses = [row.strip() for row in costs_df[costs_df.columns[0]].tolist()]
stores = [col.strip() for col in costs_df.columns[1:]]
cost = {}
for i, wh in enumerate(warehouses):
    cost[wh] = {}
    for j, store in enumerate(stores):
        val = float(costs_df.iloc[i, j + 1])
        cost[wh][store] = val
if set(demand.keys()) != set(stores):
    raise ValueError('Mismatch between demand stores and cost matrix columns')
if set(supply_capacity.keys()) != set(warehouses):
    raise ValueError('Mismatch between supply warehouses and cost matrix rows')
for wh in warehouses:
    if set(cost[wh].keys()) != set(stores):
        raise ValueError(f'Cost row for {wh} does not match store list')
m = gp.Model('Original_RAG_TP')
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