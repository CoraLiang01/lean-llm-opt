import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', dtype=str, keep_default_na=False, skipinitialspace=True, delim_whitespace=False)
demand = {}
for idx, row in demand_df.iterrows():
    key = row[demand_df.columns[0]].strip()
    val = float(row[demand_df.columns[1]])
    demand[key] = val
supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', dtype=str, keep_default_na=False, skipinitialspace=True, delim_whitespace=False)
supply_capacity = {}
for idx, row in supply_df.iterrows():
    key = row[supply_df.columns[0]].strip()
    val = float(row[supply_df.columns[1]])
    supply_capacity[key] = val
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', dtype=str, keep_default_na=False)
warehouses = [row.strip() for row in costs_df[costs_df.columns[0]].tolist()]
stores = [col.strip() for col in costs_df.columns[1:]]
cost = {}
for i, row in costs_df.iterrows():
    warehouse = row[costs_df.columns[0]].strip()
    cost[warehouse] = {}
    for store in stores:
        val = float(row[store])
        cost[warehouse][store] = val
if set(demand.keys()) != set(stores):
    raise ValueError(f'Demand store keys {set(demand.keys())} do not match cost matrix columns {set(stores)}')
if set(supply_capacity.keys()) != set(warehouses):
    raise ValueError(f'Supply warehouse keys {set(supply_capacity.keys())} do not match cost matrix rows {set(warehouses)}')
for i in warehouses:
    for j in stores:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for warehouse {i}, store {j}')
m = gp.Model('GreenMart_Transportation')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) >= demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= supply_capacity[i] for i in warehouses), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')