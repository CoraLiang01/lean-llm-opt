import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
demand_df = pd.read_csv(demand_path, sep=',', dtype=str, keep_default_na=False)
plant_ids = cost_df['plant'].astype(str).tolist()
customer_ids = demand_df['customer'].astype(str).tolist()
fixed_cost = {}
capacity = {}
for (idx, row) in cost_df.iterrows():
    plant = str(row['plant'])
    try:
        fixed_cost[plant] = float(row['fixed_cost'])
        capacity[plant] = float(row['capacity'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in cost.csv for plant {plant}: {e}')
transport_cost = {}
for (idx, row) in cost_df.iterrows():
    plant = str(row['plant'])
    for cust in customer_ids:
        try:
            transport_cost[plant, cust] = float(row[cust])
        except Exception as e:
            raise ValueError(f'Missing or invalid transport cost for plant {plant}, customer {cust}: {e}')
demand = {}
for (idx, row) in demand_df.iterrows():
    cust = str(row['customer'])
    try:
        demand[cust] = float(row['demand'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in demand.csv for customer {cust}: {e}')
if set(plant_ids) != set(fixed_cost.keys()) or set(plant_ids) != set(capacity.keys()):
    raise ValueError('Mismatch in plant identifiers between cost.csv and extracted parameter dictionaries.')
if set(customer_ids) != set(demand.keys()):
    raise ValueError('Mismatch in customer identifiers between demand.csv and extracted parameter dictionaries.')
for plant in plant_ids:
    for cust in customer_ids:
        if (plant, cust) not in transport_cost:
            raise ValueError(f'Missing transport cost for plant {plant}, customer {cust}.')
m = gp.Model('CapacitatedFacilityLocation')
x_vars = m.addVars(plant_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(plant_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in plant_ids)) + gp.quicksum((transport_cost[i, j] * x_vars[i, j] for i in plant_ids for j in customer_ids)), gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plant_ids)) == demand[j], name='')
for i in plant_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customer_ids)) <= capacity[i] * y_vars[i], name='')
m.optimize()