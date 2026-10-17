import pandas as pd
import gurobipy as gp
from gurobipy import GRB
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
if df_demand.shape[1] == 2:
    customer_ids = df_demand.iloc[:, 0].tolist()
    demand = {row[0]: float(row[1]) for row in df_demand.values}
else:
    raise ValueError('customer_demand.csv must have two columns: customer_id, demand')
df_supply = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
if df_supply.shape[1] == 2:
    supplier_ids = df_supply.iloc[:, 0].tolist()
    supply_capacity = {row[0]: float(row[1]) for row in df_supply.values}
else:
    raise ValueError('supply_capacity.csv must have two columns: supplier_id, capacity')
df_cost = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False, index_col=0)
cost_suppliers = df_cost.index.tolist()
cost_customers = df_cost.columns.tolist()
if set(supplier_ids) != set(cost_suppliers):
    raise ValueError('Supplier IDs in supply_capacity.csv and transportation_costs.csv do not match')
if set(customer_ids) != set(cost_customers):
    raise ValueError('Customer IDs in customer_demand.csv and transportation_costs.csv do not match')
cost = {i: {j: float(df_cost.loc[i, j]) for j in customer_ids} for i in supplier_ids}
m = gp.Model('Original_RAG_TP')
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