import gurobipy as gp
import pandas as pd
import numpy as np

def solve_brewco_transportation():
    demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv', dtype=str, keep_default_na=False)
    supply_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv', dtype=str, keep_default_na=False)
    cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv', dtype=str, keep_default_na=False)
    plants = supply_df['Unnamed: 0'].str.strip().unique().tolist()
    customers = demand_df['customer'].str.strip().unique().tolist()
    demand_dict = {}
    for (_, row) in demand_df.iterrows():
        cust = str(row['customer']).strip()
        try:
            demand = int(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
        demand_dict[cust] = demand
    supply_dict = {}
    for (_, row) in supply_df.iterrows():
        plant = str(row['Unnamed: 0']).strip()
        try:
            cap = int(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for plant {plant}: {row['supply_capacity']}")
        supply_dict[plant] = cap
    cost_dict = {}
    cost_plants = cost_df['Unnamed: 0'].str.strip().tolist()
    cost_customers = [col for col in cost_df.columns if col != 'Unnamed: 0']
    missing_plants = set(plants) - set(cost_plants)
    if missing_plants:
        raise ValueError(f'Plants missing in transportation_costs.csv: {missing_plants}')
    missing_customers = set(customers) - set(cost_customers)
    if missing_customers:
        raise ValueError(f'Customers missing in transportation_costs.csv: {missing_customers}')
    for (_, row) in cost_df.iterrows():
        plant = str(row['Unnamed: 0']).strip()
        for cust in customers:
            try:
                cost = float(row[cust])
            except Exception:
                raise ValueError(f'Invalid cost for plant {plant}, customer {cust}: {row[cust]}')
            cost_dict[plant, cust] = cost
    for s in plants:
        for c in customers:
            if (s, c) not in cost_dict:
                raise ValueError(f'Missing transportation cost for plant {s}, customer {c}')
    m = gp.Model('brewco_transportation')
    shipment_keys = [(s, c) for s in plants for c in customers]
    shipment_vars = m.addVars(shipment_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost_dict[s, c] * shipment_vars[s, c] for (s, c) in shipment_keys)), gp.GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((shipment_vars[s, c] for s in plants)) == demand_dict[c], name='')
    for s in plants:
        m.addConstr(gp.quicksum((shipment_vars[s, c] for c in customers)) <= supply_dict[s], name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_brewco_transportation()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')