import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
demand_df = pd.read_csv(demand_path, sep=',', dtype=str, keep_default_na=False)
cost_df['plant'] = cost_df['plant'].str.strip()
plants = cost_df['plant'].tolist()
customer_cols = [col for col in cost_df.columns if col.startswith('C')]
customers = [c.strip() for c in customer_cols]
demand_df['customer'] = demand_df['customer'].str.strip()
demand_customers = demand_df['customer'].tolist()
if set(customers) != set(demand_customers):
    raise ValueError(f'Mismatch between customers in cost.csv columns and demand.csv: {set(customers)} vs {set(demand_customers)}')
customers = customer_cols
fixed_cost = {}
capacity = {}
transport_cost = {}
for (idx, row) in cost_df.iterrows():
    plant = row['plant']
    try:
        fixed_cost[plant] = int(row['fixed_cost'])
        capacity[plant] = int(row['capacity'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in cost.csv for plant {plant}: {e}')
    for cust in customers:
        try:
            transport_cost[plant, cust] = float(row[cust])
        except Exception as e:
            raise ValueError(f'Invalid transport cost for plant {plant}, customer {cust}: {e}')
demand = {}
for (idx, row) in demand_df.iterrows():
    cust = row['customer']
    try:
        demand[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f'Invalid demand value for customer {cust}: {e}')
if set(plants) != set(cost_df['plant']):
    raise ValueError('Mismatch in plant identifiers.')
if set(customers) != set(demand.keys()):
    raise ValueError('Mismatch in customer identifiers.')
m = gp.Model('UFLP')
x_keys = [(i, j) for i in plants for j in customers]
x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(plants, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in plants)) + gp.quicksum((transport_cost[i, j] * x_vars[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')