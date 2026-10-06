import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_transportation_problem():
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
    df_demand = pd.read_csv(demand_path, sep=',')
    df_demand['Customers'] = df_demand['Customers'].astype(str).str.strip()
    customers = list(df_demand['Customers'])
    demand_dict = dict(zip(df_demand['Customers'], df_demand['demand']))
    df_supply = pd.read_csv(supply_path, sep=',')
    df_supply['Supplier'] = df_supply['Supplier'].astype(str).str.strip()
    suppliers = list(df_supply['Supplier'])
    supply_dict = dict(zip(df_supply['Supplier'], df_supply['supply_capacity']))
    df_cost = pd.read_csv(cost_path, sep=',')
    df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].astype(str).str.strip()
    cost_suppliers = list(df_cost['Unnamed: 0'])
    cost_customers = [col for col in df_cost.columns if col != 'Unnamed: 0']
    missing_suppliers = set(suppliers) - set(cost_suppliers)
    missing_customers = set(customers) - set(cost_customers)
    if missing_suppliers:
        raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
    if missing_customers:
        raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
    cost_dict = {}
    for _, row in df_cost.iterrows():
        s = row['Unnamed: 0']
        for c in customers:
            cost_dict[s, c] = float(row[c])
    m = gp.Model('TransportationProblem')
    x = m.addVars(suppliers, customers, lb=0.0, name='')
    m.setObjective(gp.quicksum((cost_dict[s, c] * x[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[s, c] for s in suppliers)) == demand_dict[c] for c in customers), name='')
    m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= supply_dict[s] for s in suppliers), name='')
    m.optimize()
    return m
m = solve_transportation_problem()