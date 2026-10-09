import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
customer_df = pd.read_csv(customer_demand_path, dtype=str, keep_default_na=False)
customer_df['demand'] = customer_df['demand'].astype(float)
customers = customer_df['Customers'].str.strip().tolist()
customer_demand = dict(zip(customers, customer_df['demand']))
supply_df = pd.read_csv(supply_capacity_path, dtype=str, keep_default_na=False)
supply_df['supply_capacity'] = supply_df['supply_capacity'].astype(float)
suppliers = supply_df['Supplier'].str.strip().tolist()
supply_capacity = dict(zip(suppliers, supply_df['supply_capacity']))
costs_df = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
costs_df['Unnamed: 0'] = costs_df['Unnamed: 0'].str.strip()
costs_df = costs_df.set_index('Unnamed: 0')
cost_supplier_set = set(costs_df.index)
cost_customer_set = set(costs_df.columns)
if set(suppliers) != cost_supplier_set:
    raise ValueError(f'Mismatch in suppliers between supply_capacity.csv and transportation_costs.csv: {set(suppliers)} vs {cost_supplier_set}')
if set(customers) != cost_customer_set:
    raise ValueError(f'Mismatch in customers between customer_demand.csv and transportation_costs.csv: {set(customers)} vs {cost_customer_set}')
cost = {}
for s in suppliers:
    for c in customers:
        val = costs_df.loc[s, c]
        try:
            cost[s, c] = float(val)
        except Exception:
            raise ValueError(f'Invalid cost value for supplier {s}, customer {c}: {val}')
if set(customer_demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys and customer list.')
if set(supply_capacity.keys()) != set(suppliers):
    raise ValueError('Mismatch in supply capacity keys and supplier list.')
shipment_keys = [(s, c) for s in suppliers for c in customers]
m = gp.Model('TransportationProblem')
x_vars = m.addVars(shipment_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[s, c] * x_vars[s, c] for (s, c) in shipment_keys)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == customer_demand[c], name=f'demand_{c}')
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for (s, c) in shipment_keys:
        print(f'{x_vars[s, c].VarName} {x_vars[s, c].X}')
else:
    print(f'Solver status: {m.Status}')