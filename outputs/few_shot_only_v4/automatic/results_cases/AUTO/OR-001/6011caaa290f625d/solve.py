import gurobipy as gp
from gurobipy import GRB
import pandas as pd
customer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv', dtype=str, keep_default_na=False)
supply_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv', dtype=str, keep_default_na=False)
transportation_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv', dtype=str, keep_default_na=False)
if customer_demand_df.shape[1] == 2:
    customer_col = customer_demand_df.columns[0]
    demand_col = customer_demand_df.columns[1]
    customers = customer_demand_df[customer_col].tolist()
    demand = {row[customer_col]: float(row[demand_col]) for _, row in customer_demand_df.iterrows()}
else:
    raise ValueError('customer_demand.csv must have exactly two columns (customer, demand)')
if supply_capacity_df.shape[1] == 2:
    supplier_col = supply_capacity_df.columns[0]
    supply_col = supply_capacity_df.columns[1]
    suppliers = supply_capacity_df[supplier_col].tolist()
    supply_capacity = {row[supplier_col]: float(row[supply_col]) for _, row in supply_capacity_df.iterrows()}
else:
    raise ValueError('supply_capacity.csv must have exactly two columns (supplier, supply_capacity)')
cost = {}
cost_suppliers = list(transportation_costs_df.iloc[:, 0])
cost_customers = list(transportation_costs_df.columns[1:])
if set(cost_suppliers) != set(suppliers):
    raise ValueError('Supplier identifiers in transportation_costs.csv do not match supply_capacity.csv')
if set(cost_customers) != set(customers):
    raise ValueError('Customer identifiers in transportation_costs.csv do not match customer_demand.csv')
for i, row in transportation_costs_df.iterrows():
    supplier = row.iloc[0]
    cost[supplier] = {}
    for j, customer in enumerate(cost_customers):
        val = row.iloc[j + 1]
        try:
            cost[supplier][customer] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for supplier {supplier}, customer {customer}: {val}')
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
m = gp.Model('Amazon_Distribution_Transportation')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')