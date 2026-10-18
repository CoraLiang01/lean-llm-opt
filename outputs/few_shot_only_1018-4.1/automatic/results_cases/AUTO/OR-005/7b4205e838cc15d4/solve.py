import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv', dtype=str, keep_default_na=False, skipinitialspace=True)
demand_df.columns = [c.strip() for c in demand_df.columns]
demand = {}
for idx, row in demand_df.iterrows():
    cust = row[demand_df.columns[0]].strip()
    val = float(row[demand_df.columns[1]])
    demand[cust] = val
supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv', dtype=str, keep_default_na=False, skipinitialspace=True)
supply_df.columns = [c.strip() for c in supply_df.columns]
supply_capacity = {}
for idx, row in supply_df.iterrows():
    supplier = row[supply_df.columns[0]].strip()
    val = float(row[supply_df.columns[1]])
    supply_capacity[supplier] = val
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv', dtype=str, keep_default_na=False, skipinitialspace=True)
costs_df.columns = [c.strip() for c in costs_df.columns]
suppliers = []
customers = [c.strip() for c in costs_df.columns[1:]]
cost = {}
for idx, row in costs_df.iterrows():
    supplier = row[costs_df.columns[0]].strip()
    suppliers.append(supplier)
    cost[supplier] = {}
    for j, cust in enumerate(customers):
        val = float(row[costs_df.columns[j + 1]])
        cost[supplier][cust] = val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch between customer demand and cost matrix columns')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch between supply capacity and cost matrix rows')
for i in suppliers:
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for supplier {i}, customer {j}')
m = gp.Model('Amazon_Distribution_Transportation')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')