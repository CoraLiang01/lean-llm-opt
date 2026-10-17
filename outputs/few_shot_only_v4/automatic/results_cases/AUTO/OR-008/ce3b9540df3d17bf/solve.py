import gurobipy as gp
from gurobipy import GRB
import pandas as pd
customer_demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv', dtype=str, keep_default_na=False)
if customer_demand_df.shape[1] == 2:
    customer_ids = customer_demand_df.iloc[:, 0].tolist()
    demand = {row[0]: float(row[1]) for row in customer_demand_df.values}
else:
    raise ValueError('customer_demand.csv must have two columns: customer_id, demand')
supply_capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv', dtype=str, keep_default_na=False)
if supply_capacity_df.shape[1] == 2:
    supplier_ids = supply_capacity_df.iloc[:, 0].tolist()
    supply_capacity = {row[0]: float(row[1]) for row in supply_capacity_df.values}
else:
    raise ValueError('supply_capacity.csv must have two columns: supplier_id, capacity')
transportation_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv', dtype=str, keep_default_na=False, index_col=0)
cost_suppliers = list(transportation_costs_df.index)
cost_customers = list(transportation_costs_df.columns)
cost = {}
for i in cost_suppliers:
    cost[i] = {}
    for j in cost_customers:
        try:
            cost[i][j] = float(transportation_costs_df.loc[i, j])
        except Exception:
            raise ValueError(f'Missing or invalid cost for supplier {i}, customer {j}')
if set(supplier_ids) != set(cost_suppliers):
    raise ValueError('Mismatch between suppliers in supply_capacity.csv and transportation_costs.csv')
if set(customer_ids) != set(cost_customers):
    raise ValueError('Mismatch between customers in customer_demand.csv and transportation_costs.csv')
m = gp.Model('FreshMart_Transportation')
x_vars = m.addVars(supplier_ids, customer_ids, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in supplier_ids for j in customer_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in supplier_ids)) >= demand[j] for j in customer_ids), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customer_ids)) <= supply_capacity[i] for i in supplier_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')