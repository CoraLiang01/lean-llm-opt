import gurobipy as gp
import pandas as pd
import numpy as np
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_cost_path, sep=',')
customers = [str(c).strip() for c in demand_df['customer']]
warehouses = [str(w).strip() for w in fixed_cost_df['Unnamed: 0']]
transport_warehouses = [str(w).strip() for w in transport_df['Unnamed: 0']]
transport_customers = [str(c).strip() for c in transport_df.columns if c != 'Unnamed: 0']
if set(warehouses) != set(transport_warehouses):
    raise ValueError('Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set(transport_customers):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
warehouses = sorted(warehouses)
customers = sorted(customers)
fixed_cost = {}
for _, row in fixed_cost_df.iterrows():
    w = str(row['Unnamed: 0']).strip()
    fixed_cost[w] = float(row['fixed_costs'])
demand = {}
for _, row in demand_df.iterrows():
    c = str(row['customer']).strip()
    demand[c] = int(row['demand'])
transport_cost = {}
transport_df_indexed = transport_df.set_index('Unnamed: 0')
for w in warehouses:
    for c in customers:
        cost = float(transport_df_indexed.loc[w, c])
        transport_cost[w, c] = cost
m = gp.Model('UFLP_Bandcamp')
x = m.addVars(warehouses, customers, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[w] * y[w] for w in warehouses)) + gp.quicksum((transport_cost[w, c] * x[w, c] for w in warehouses for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[w, c] for w in warehouses)) == demand[c], name=f'demand_{c}')
for w in warehouses:
    for c in customers:
        m.addConstr(x[w, c] <= demand[c] * y[w], name=f'link_{w}_{c}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Warehouse Activation ---')
    for w in warehouses:
        print(f"Warehouse {w}: {('OPEN' if y[w].X > 0.5 else 'CLOSED')} (y={int(round(y[w].X))})")
    print('\n--- Shipment Plan (x[w, c] > 0) ---')
    for w in warehouses:
        for c in customers:
            if x[w, c].X > 1e-06:
                print(f'  {x[w, c].X:.2f} units from warehouse {w} to customer {c}')
else:
    print(f'No optimal solution found. Status: {m.status}')