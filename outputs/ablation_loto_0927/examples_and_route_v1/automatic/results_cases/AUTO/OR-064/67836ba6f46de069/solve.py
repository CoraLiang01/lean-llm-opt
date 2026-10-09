import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = list(demand_df['customer'])
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = list(fixed_cost_df['supplier'])
fixed_cost_dict = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['supplier'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
trans_cost_df.columns = [col.strip() if col != 'Unnamed: 0' else col for col in trans_cost_df.columns]
if set(suppliers) != set(trans_cost_df['supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) - set([col for col in trans_cost_df.columns if col.startswith('C')]):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
transport_cost = {}
for (_, row) in trans_cost_df.iterrows():
    supplier = row['supplier']
    for customer in customers:
        if customer not in row:
            raise ValueError(f'Customer {customer} missing in transportation_costs.csv for supplier {supplier}')
        transport_cost[supplier, customer] = float(row[customer])
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y[i] for i in suppliers))
transport_cost_expr = gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= demand_dict[j] * y[i], name=f'activate_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Opening Decisions ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'closed')} (y={int(round(y[i].X))})")
    print('\n--- Supply Plan (nonzero flows) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')