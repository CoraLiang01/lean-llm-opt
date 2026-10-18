import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, header=None, skiprows=1)
customers = []
demand = {}
for idx, row in demand_df.iterrows():
    cust = row[0].strip()
    val = float(row[1])
    customers.append(cust)
    demand[cust] = val
supply_df = pd.read_csv(supply_path, dtype=str, keep_default_na=False)
suppliers = []
supply_capacity = {}
for idx, row in supply_df.iterrows():
    store = row['Unnamed: 0'].strip()
    val = float(row['supply_capacity'])
    suppliers.append(store)
    supply_capacity[store] = val
cost_df = pd.read_csv(cost_path, dtype=str, keep_default_na=False)
cost_df.set_index('Unnamed: 0', inplace=True)
cost = {}
for store in suppliers:
    if store not in cost_df.index:
        raise ValueError(f'Store {store} missing in cost matrix')
    cost[store] = {}
    for cust in customers:
        if cust not in cost_df.columns:
            raise ValueError(f'Customer {cust} missing in cost matrix')
        val = float(cost_df.loc[store, cust])
        cost[store][cust] = val
if set(cost.keys()) != set(suppliers):
    raise ValueError('Mismatch in suppliers between cost and supply')
for store in suppliers:
    if set(cost[store].keys()) != set(customers):
        raise ValueError(f'Mismatch in customers for store {store} in cost matrix')
m = gp.Model('Walmart_Transportation')
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