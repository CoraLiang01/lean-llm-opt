import pandas as pd
import gurobipy as gp
from gurobipy import GRB
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv', dtype=str, keep_default_na=False)
customers = demand_df['customer'].tolist()
demand = {}
for idx, row in demand_df.iterrows():
    cust = row['customer'].strip()
    try:
        val = float(row['demand'])
    except Exception:
        raise ValueError(f'Invalid demand value for customer {cust}')
    demand[cust] = val
supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv', dtype=str, keep_default_na=False)
plants = supply_df['Unnamed: 0'].tolist()
supply_capacity = {}
for idx, row in supply_df.iterrows():
    plant = row['Unnamed: 0'].strip()
    try:
        val = float(row['supply_capacity'])
    except Exception:
        raise ValueError(f'Invalid supply capacity for plant {plant}')
    supply_capacity[plant] = val
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', dtype=str, keep_default_na=False)
costs_df.set_index('Unnamed: 0', inplace=True)
cost = {}
for plant in plants:
    if plant not in costs_df.index:
        raise ValueError(f'Missing cost row for plant {plant}')
    cost[plant] = {}
    for cust in customers:
        if cust not in costs_df.columns:
            raise ValueError(f'Missing cost column for customer {cust}')
        try:
            val = float(costs_df.loc[plant, cust])
        except Exception:
            raise ValueError(f'Invalid cost for plant {plant}, customer {cust}')
        cost[plant][cust] = val
if set(demand.keys()) != set(customers):
    raise ValueError('Mismatch in customer demand keys and customer list')
if set(supply_capacity.keys()) != set(plants):
    raise ValueError('Mismatch in supply capacity keys and plant list')
for plant in plants:
    if set(cost[plant].keys()) != set(customers):
        raise ValueError(f'Mismatch in cost columns for plant {plant}')
m = gp.Model('BrewCo_Transportation')
x_vars = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[plant][cust] * x_vars[plant, cust] for plant in plants for cust in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[plant, cust] for plant in plants)) >= demand[cust] for cust in customers), name='')
m.addConstrs((gp.quicksum((x_vars[plant, cust] for cust in customers)) <= supply_capacity[plant] for plant in plants), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')