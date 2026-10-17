import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
if demand_df.shape[1] == 2:
    customers = demand_df.iloc[:, 0].tolist()
    demand = {row[0]: float(row[1]) for row in demand_df.values}
else:
    raise ValueError('customer_demand.csv must have exactly two columns (customer, demand)')
supply_df = pd.read_csv(supply_path, dtype=str, keep_default_na=False)
if supply_df.shape[1] == 2:
    suppliers = supply_df.iloc[:, 0].tolist()
    supply_capacity = {row[0]: float(row[1]) for row in supply_df.values}
else:
    raise ValueError('supply_capacity.csv must have exactly two columns (supplier, supply)')
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
if cost_df.shape[1] < 2:
    raise ValueError('transportation_costs.csv must have at least two columns (supplier, ...customers)')
cost_suppliers = cost_df.iloc[:, 0].tolist()
cost_customers = list(cost_df.columns[1:])
if set(suppliers) != set(cost_suppliers):
    raise ValueError('Mismatch between suppliers in supply_capacity.csv and transportation_costs.csv')
if set(customers) != set(cost_customers):
    raise ValueError('Mismatch between customers in customer_demand.csv and transportation_costs.csv')
cost_df = cost_df.set_index(cost_df.columns[0])
cost_df = cost_df.loc[suppliers, cost_customers]
cost = {i: {j: float(cost_df.loc[i, j]) for j in cost_customers} for i in suppliers}
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
for i in suppliers:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply for supplier {i}')
for i in suppliers:
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for supplier {i}, customer {j}')
m = gp.Model('Original_RAG_TP')
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