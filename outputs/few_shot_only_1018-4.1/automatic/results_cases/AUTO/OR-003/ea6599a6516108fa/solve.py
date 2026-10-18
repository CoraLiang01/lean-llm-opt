import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv', dtype=str, keep_default_na=False)
customers = []
demand = {}
for idx, row in demand_df.iterrows():
    cust = row['customer'].strip()
    customers.append(cust)
    try:
        demand[cust] = float(row['demand'])
    except Exception:
        raise ValueError(f'Invalid demand value for customer {cust}')
supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv', dtype=str, keep_default_na=False)
suppliers = []
supply_capacity = {}
for idx, row in supply_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    suppliers.append(sup)
    try:
        supply_capacity[sup] = float(row['supply_capacity'])
    except Exception:
        raise ValueError(f'Invalid supply capacity for supplier {sup}')
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv', dtype=str, keep_default_na=False)
cost = {}
for idx, row in costs_df.iterrows():
    sup = row['Unnamed: 0'].strip()
    if sup not in suppliers:
        raise ValueError(f'Supplier {sup} in costs not found in supply_capacity.csv')
    cost[sup] = {}
    for cust in customers:
        try:
            cost_val = float(row[cust])
        except Exception:
            raise ValueError(f'Invalid cost for supplier {sup}, customer {cust}')
        cost[sup][cust] = cost_val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys and customer list')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch in supply capacity keys and supplier list')
for sup in suppliers:
    if sup not in cost:
        raise ValueError(f'Missing cost row for supplier {sup}')
    for cust in customers:
        if cust not in cost[sup]:
            raise ValueError(f'Missing cost entry for supplier {sup}, customer {cust}')
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