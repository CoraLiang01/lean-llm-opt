import gurobipy as gp
import pandas as pd
import numpy as np
plant_capacity_path = 'benchmark_dataset/Variant12/inputs/plant_capacity.csv'
retailer_demand_path = 'benchmark_dataset/Variant12/inputs/retailer_demand.csv'
route_variable_costs_path = 'benchmark_dataset/Variant12/inputs/route_variable_costs.csv'
route_fixed_costs_path = 'benchmark_dataset/Variant12/inputs/route_fixed_costs.csv'
plant_df = pd.read_csv(plant_capacity_path, dtype=str, keep_default_na=False)
retailer_df = pd.read_csv(retailer_demand_path, dtype=str, keep_default_na=False)
varcost_df = pd.read_csv(route_variable_costs_path, dtype=str, keep_default_na=False)
fixedcost_df = pd.read_csv(route_fixed_costs_path, dtype=str, keep_default_na=False)
plant_df['Plant'] = plant_df['Plant'].str.strip()
plants = plant_df['Plant'].tolist()
plant_cap = {}
for (_, row) in plant_df.iterrows():
    plant = row['Plant']
    try:
        cap = int(row['SupplyCapacity'])
    except Exception:
        raise ValueError(f"Invalid SupplyCapacity for Plant {plant}: {row['SupplyCapacity']}")
    plant_cap[plant] = cap
retailer_df['Retailer'] = retailer_df['Retailer'].str.strip()
retailers = retailer_df['Retailer'].tolist()
retailer_dem = {}
for (_, row) in retailer_df.iterrows():
    retailer = row['Retailer']
    try:
        dem = int(row['Demand'])
    except Exception:
        raise ValueError(f"Invalid Demand for Retailer {retailer}: {row['Demand']}")
    retailer_dem[retailer] = dem
varcost_df['Plant'] = varcost_df['Plant'].str.strip()
if not set(plants).issubset(set(varcost_df['Plant'])):
    raise ValueError('Mismatch between plants in plant_capacity.csv and route_variable_costs.csv')
if not set(retailers).issubset(set(varcost_df.columns[1:])):
    raise ValueError('Mismatch between retailers in retailer_demand.csv and route_variable_costs.csv')
c = {}
for (_, row) in varcost_df.iterrows():
    plant = row['Plant']
    for retailer in retailers:
        try:
            c[plant, retailer] = float(row[retailer])
        except Exception:
            raise ValueError(f'Invalid variable cost for route ({plant},{retailer}): {row[retailer]}')
fixedcost_df['Plant'] = fixedcost_df['Plant'].str.strip()
if not set(plants).issubset(set(fixedcost_df['Plant'])):
    raise ValueError('Mismatch between plants in plant_capacity.csv and route_fixed_costs.csv')
if not set(retailers).issubset(set(fixedcost_df.columns[1:])):
    raise ValueError('Mismatch between retailers in retailer_demand.csv and route_fixed_costs.csv')
f = {}
for (_, row) in fixedcost_df.iterrows():
    plant = row['Plant']
    for retailer in retailers:
        try:
            f[plant, retailer] = float(row[retailer])
        except Exception:
            raise ValueError(f'Invalid fixed cost for route ({plant},{retailer}): {row[retailer]}')
M = {}
for i in plants:
    for j in retailers:
        M[i, j] = min(plant_cap[i], retailer_dem[j])
m = gp.Model('FixedChargeTransportation')
x_vars = m.addVars(plants, retailers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(plants, retailers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] + f[i, j] * y_vars[i, j] for i in plants for j in retailers)), gp.GRB.MINIMIZE)
for j in retailers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in plants)) == retailer_dem[j], name=f'demand_{j}')
for i in plants:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in retailers)) <= plant_cap[i], name=f'supply_{i}')
for i in plants:
    for j in retailers:
        m.addConstr(x_vars[i, j] <= M[i, j] * y_vars[i, j], name=f'link_{i}_{j}')
m.optimize()