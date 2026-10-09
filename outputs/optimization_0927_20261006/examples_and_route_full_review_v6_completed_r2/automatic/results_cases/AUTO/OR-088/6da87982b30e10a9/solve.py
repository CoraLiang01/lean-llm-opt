import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
demand_df = pd.read_csv(demand_path, sep=',', dtype=str, keep_default_na=False)
plants = cost_df['plant'].astype(str).tolist()
customer_cols = [col for col in cost_df.columns if col.startswith('C')]
customers = demand_df['customer'].astype(str).tolist()
if set(customers) != set(customer_cols):
    raise ValueError(f'Customer columns in cost.csv ({customer_cols}) do not match customers in demand.csv ({customers})')
fixed_cost = {}
capacity = {}
transport_cost = {}
for (idx, row) in cost_df.iterrows():
    plant = str(row['plant'])
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
    cust = str(row['customer'])
    try:
        demand[cust] = int(row['demand'])
    except Exception as e:
        raise ValueError(f'Invalid demand value for customer {cust}: {e}')
m = gp.Model('CapacitatedFacilityLocation')
y_vars = m.addVars(plants, vtype=gp.GRB.BINARY, name='')
x_vars = m.addVars(plants, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in plants)) + gp.quicksum((transport_cost[i, j] * x_vars[i, j] for i in plants for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == demand[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= capacity[i] * y_vars[i], name=f'capacity_{i}')
m.optimize()