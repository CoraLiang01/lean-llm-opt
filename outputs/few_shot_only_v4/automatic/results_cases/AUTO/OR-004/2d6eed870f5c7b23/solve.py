import pandas as pd
import gurobipy as gp
from gurobipy import GRB
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
customer_demand_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
if customer_demand_df.shape[1] == 2:
    customer_col = customer_demand_df.columns[0]
    demand_col = customer_demand_df.columns[1]
    customers = customer_demand_df[customer_col].tolist()
    demand = {}
    for _, row in customer_demand_df.iterrows():
        try:
            demand[row[customer_col]] = float(row[demand_col])
        except Exception:
            raise ValueError(f'Invalid demand value for customer {row[customer_col]}')
else:
    raise ValueError('customer_demand.csv must have exactly two columns (customer, demand)')
if supply_capacity_df.shape[1] == 2:
    supplier_col = supply_capacity_df.columns[0]
    capacity_col = supply_capacity_df.columns[1]
    suppliers = supply_capacity_df[supplier_col].tolist()
    supply_capacity = {}
    for _, row in supply_capacity_df.iterrows():
        try:
            supply_capacity[row[supplier_col]] = float(row[capacity_col])
        except Exception:
            raise ValueError(f'Invalid supply capacity value for supplier {row[supplier_col]}')
else:
    raise ValueError('supply_capacity.csv must have exactly two columns (supplier, capacity)')
cost = {}
cost_customers = list(transportation_costs_df.columns)[1:]
if set(cost_customers) != set(customers):
    raise ValueError('Mismatch between customer groups in transportation_costs.csv and customer_demand.csv')
for _, row in transportation_costs_df.iterrows():
    supplier = row[transportation_costs_df.columns[0]]
    if supplier not in suppliers:
        raise ValueError(f'Supplier {supplier} in transportation_costs.csv not found in supply_capacity.csv')
    cost[supplier] = {}
    for customer in cost_customers:
        try:
            cost[supplier][customer] = float(row[customer])
        except Exception:
            raise ValueError(f'Invalid cost value for supplier {supplier}, customer {customer}')
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer identifiers between demand and customers list')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch in supplier identifiers between supply_capacity and suppliers list')
for i in suppliers:
    if i not in cost:
        raise ValueError(f'Missing cost row for supplier {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost entry for supplier {i}, customer {j}')
m = gp.Model('Optimal_Fulfillment_Policy')
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