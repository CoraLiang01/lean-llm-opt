import pandas as pd
import gurobipy as gp
from gurobipy import GRB
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if customer_demand_df.shape[1] == 2:
    customer_col = customer_demand_df.columns[0]
    demand_col = customer_demand_df.columns[1]
    customers = customer_demand_df[customer_col].tolist()
    demand = {row[customer_col]: float(row[demand_col]) for _, row in customer_demand_df.iterrows()}
else:
    raise ValueError('customer_demand.csv must have exactly two columns (customer, demand)')
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if supply_capacity_df.shape[1] == 2:
    supplier_col = supply_capacity_df.columns[0]
    capacity_col = supply_capacity_df.columns[1]
    suppliers = supply_capacity_df[supplier_col].tolist()
    supply_capacity = {row[supplier_col]: float(row[capacity_col]) for _, row in supply_capacity_df.iterrows()}
else:
    raise ValueError('supply_capacity.csv must have exactly two columns (supplier, capacity)')
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if transportation_costs_df.shape[0] != len(suppliers):
    raise ValueError('Row count of transportation_costs.csv does not match number of suppliers')
if transportation_costs_df.shape[1] != len(customers):
    raise ValueError('Column count of transportation_costs.csv does not match number of customers')
cost = {}
for i, supplier in enumerate(suppliers):
    cost[supplier] = {}
    for j, customer in enumerate(customers):
        val = transportation_costs_df.iloc[i, j]
        try:
            cost[supplier][customer] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value at supplier {supplier}, customer {customer}: {val}')
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch between customer_demand.csv and transportation_costs.csv customer identifiers')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch between supply_capacity.csv and transportation_costs.csv supplier identifiers')
for supplier in suppliers:
    if set(cost[supplier].keys()) != set(customers):
        raise ValueError(f'Cost data for supplier {supplier} does not cover all customers')
m = gp.Model('Amazon_Distribution_TP')
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