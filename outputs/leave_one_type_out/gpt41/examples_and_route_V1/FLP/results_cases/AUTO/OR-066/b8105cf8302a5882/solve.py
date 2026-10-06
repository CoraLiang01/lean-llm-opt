import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', sep=',')
suppliers_fc = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
suppliers_tc = trans_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
if set(suppliers_fc) != set(suppliers_tc):
    raise ValueError(f'Supplier sets in fixed_cost.csv and transportation_costs.csv do not match: {suppliers_fc} vs {suppliers_tc}')
suppliers = suppliers_fc
customers_demand = demand_df['customer'].astype(str).str.strip().tolist()
customers_tc = [c for c in trans_cost_df.columns if c != 'Unnamed: 0']
if set(customers_demand) != set(customers_tc):
    raise ValueError(f'Customer sets in demand.csv and transportation_costs.csv do not match: {customers_demand} vs {customers_tc}')
customers = customers_demand
fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs']))
demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand']))
trans_cost = {}
for idx, row in trans_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    for customer in customers:
        trans_cost[supplier, customer] = float(row[customer])
if set(fixed_cost.keys()) != set(suppliers):
    raise ValueError('Mismatch in supplier keys between fixed_cost and suppliers list.')
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer keys between demand and customers list.')
for i in suppliers:
    for j in customers:
        if (i, j) not in trans_cost:
            raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
m = gp.Model('UFLP')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
obj = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((trans_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('ACTIVE' if y[i].X > 0.5 else 'inactive')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')