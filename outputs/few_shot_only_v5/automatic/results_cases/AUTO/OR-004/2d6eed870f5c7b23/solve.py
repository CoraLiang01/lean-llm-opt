import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv', dtype=str, keep_default_na=False)
customers = []
demand = {}
for idx, row in demand_df.iterrows():
    cust = row[demand_df.columns[0]].strip()
    val = float(row[demand_df.columns[1]])
    customers.append(cust)
    demand[cust] = val
supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv', dtype=str, keep_default_na=False)
suppliers = []
supply_capacity = {}
for idx, row in supply_df.iterrows():
    sup = row[supply_df.columns[0]].strip()
    val = float(row[supply_df.columns[1]])
    suppliers.append(sup)
    supply_capacity[sup] = val
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv', dtype=str, keep_default_na=False)
cost = {}
header = [h.strip() for h in costs_df.columns.tolist()]
customer_cols = header[1:]
if set(customer_cols) != set(customers):
    raise ValueError('Mismatch between customer columns in cost matrix and demand file.')
for idx, row in costs_df.iterrows():
    sup = row[header[0]].strip()
    if sup not in suppliers:
        raise ValueError(f'Supplier {sup} in cost matrix not found in supply file.')
    cost[sup] = {}
    for cust in customer_cols:
        val = float(row[cust])
        cost[sup][cust] = val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys.')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch in supply capacity keys.')
for sup in suppliers:
    if sup not in cost:
        raise ValueError(f'Missing cost row for supplier {sup}.')
    for cust in customers:
        if cust not in cost[sup]:
            raise ValueError(f'Missing cost entry for supplier {sup}, customer {cust}.')
m = gp.Model('Retail_Supply_Chain_Optimization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[sup][cust] * x_vars[sup, cust] for sup in suppliers for cust in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[sup, cust] for sup in suppliers)) >= demand[cust] for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[sup, cust] for cust in customers)) <= supply_capacity[sup] for sup in suppliers), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')