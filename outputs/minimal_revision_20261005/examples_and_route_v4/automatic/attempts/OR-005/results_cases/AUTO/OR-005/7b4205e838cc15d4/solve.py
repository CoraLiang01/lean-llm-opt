import gurobipy as gp
import pandas as pd
import numpy as np
customer_demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'

def solve_transportation_problem():
    df_demand = pd.read_csv(customer_demand_path, sep=',')
    df_demand['Customers'] = df_demand['Customers'].astype(str).str.strip()
    customers = df_demand['Customers'].tolist()
    demand_dict = dict(zip(df_demand['Customers'], df_demand['demand']))
    df_supply = pd.read_csv(supply_capacity_path, sep=',')
    df_supply['Supplier'] = df_supply['Supplier'].astype(str).str.strip()
    suppliers = df_supply['Supplier'].tolist()
    supply_dict = dict(zip(df_supply['Supplier'], df_supply['supply_capacity']))
    df_cost = pd.read_csv(transportation_costs_path, sep=',')
    df_cost['Unnamed: 0'] = df_cost['Unnamed: 0'].astype(str).str.strip()
    cost_suppliers = df_cost['Unnamed: 0'].tolist()
    cost_customers = [col for col in df_cost.columns if col != 'Unnamed: 0']
    missing_suppliers = set(suppliers) - set(cost_suppliers)
    missing_customers = set(customers) - set(cost_customers)
    if missing_suppliers:
        raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
    if missing_customers:
        raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
    cost_dict = {}
    for (_, row) in df_cost.iterrows():
        s = row['Unnamed: 0']
        for c in customers:
            val = row[c]
            if pd.isnull(val):
                raise ValueError(f'Missing cost for supplier {s}, customer {c}')
            cost_dict[s, c] = float(val)
    for s in suppliers:
        if s not in supply_dict:
            raise ValueError(f'Supplier {s} missing in supply_capacity.csv')
        for c in customers:
            if (s, c) not in cost_dict:
                raise ValueError(f'Missing cost entry for supplier {s}, customer {c}')
    for c in customers:
        if c not in demand_dict:
            raise ValueError(f'Customer {c} missing in customer_demand.csv')
    m = gp.Model('TransportationProblem')
    x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost_dict[s, c] * x[s, c] for s in suppliers for c in customers)), gp.GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand_dict[c], name='demand')
    for s in suppliers:
        m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_dict[s], name='supply')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for s in suppliers:
            for c in customers:
                var = x[s, c]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_transportation_problem()