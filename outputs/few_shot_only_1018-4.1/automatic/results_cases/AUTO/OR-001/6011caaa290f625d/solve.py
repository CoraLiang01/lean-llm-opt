import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv', dtype=str, keep_default_na=False)
customers = []
demand = {}
for idx, row in demand_df.iterrows():
    cust = row['customer'].strip()
    customers.append(cust)
    demand[cust] = float(row['demand'])
supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv', dtype=str, keep_default_na=False)
suppliers = []
supply_capacity = {}
for idx, row in supply_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    suppliers.append(sup)
    supply_capacity[sup] = float(row['supply_capacity'])
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv', dtype=str, keep_default_na=False)
cost = {}
costs_df = costs_df.set_index('Unnamed: 0')
for sup in suppliers:
    cost[sup] = {}
    for cust in customers:
        val = costs_df.loc[sup, cust]
        cost[sup][cust] = float(val)
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys and customer list')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch in supply capacity keys and supplier list')
for sup in suppliers:
    if set(cost[sup].keys()) != set(customers):
        raise ValueError(f'Cost data missing for supplier {sup}')
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