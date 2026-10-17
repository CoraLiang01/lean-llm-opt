import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv'
supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv'
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if demand_df.shape[1] == 2:
    demand_ids = demand_df.iloc[:, 0].tolist()
    demand_vals = demand_df.iloc[:, 1].astype(float).tolist()
    customers = demand_ids
    demand = dict(zip(demand_ids, demand_vals))
else:
    raise ValueError('customer_demand.csv must have two columns: id, demand')
supply_df = pd.read_csv(supply_path, dtype=str, keep_default_na=False)
if supply_df.shape[1] == 2:
    supply_ids = supply_df.iloc[:, 0].tolist()
    supply_vals = supply_df.iloc[:, 1].astype(float).tolist()
    plants = supply_ids
    supply_capacity = dict(zip(supply_ids, supply_vals))
else:
    raise ValueError('supply_capacity.csv must have two columns: id, capacity')
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
if cost_df.shape[1] < 2:
    raise ValueError('transportation_costs.csv must have at least two columns: plant id and at least one customer column')
cost_row_ids = cost_df.iloc[:, 0].tolist()
cost_col_ids = cost_df.columns[1:].tolist()
if set(cost_row_ids) != set(plants):
    raise ValueError('Mismatch between plant ids in supply_capacity.csv and transportation_costs.csv')
if set(cost_col_ids) != set(customers):
    raise ValueError('Mismatch between customer ids in customer_demand.csv and transportation_costs.csv')
cost = {}
for i, plant in enumerate(cost_row_ids):
    cost[plant] = {}
    for j, cust in enumerate(cost_col_ids):
        val = cost_df.iloc[i, j + 1]
        try:
            cost[plant][cust] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for plant {plant}, customer {cust}: {val}')
if set(plants) != set(supply_capacity.keys()):
    raise ValueError('Plant ids mismatch between supply_capacity and cost matrix')
if set(customers) != set(demand.keys()):
    raise ValueError('Customer ids mismatch between customer_demand and cost matrix')
for i in plants:
    if set(cost[i].keys()) != set(customers):
        raise ValueError(f'Cost matrix missing customers for plant {i}')
m = gp.Model('BrewCo_Transportation')
x_vars = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in plants)) >= demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i] for i in plants), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')