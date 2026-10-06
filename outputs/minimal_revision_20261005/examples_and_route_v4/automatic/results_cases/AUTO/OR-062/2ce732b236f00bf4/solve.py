import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_id(s):
    return re.sub('\\s+', ' ', str(s)).strip()

def solve_uflp():
    demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', sep=',')
    fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', sep=',')
    trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', sep=',')
    customers = [normalize_id(c) for c in demand_df['Customer']]
    suppliers_fixed = [normalize_id(s) for s in fixed_cost_df['Unnamed: 0']]
    suppliers_trans = [normalize_id(s) for s in trans_cost_df['Unnamed: 0']]
    if set(suppliers_fixed) != set(suppliers_trans):
        raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
    suppliers = suppliers_fixed
    store_cols = [c for c in trans_cost_df.columns if c != 'Unnamed: 0']
    stores = [normalize_id(c) for c in store_cols]
    demand_dict = {}
    for (idx, row) in demand_df.iterrows():
        cust = normalize_id(row['Customer'])
        if cust in demand_dict:
            raise ValueError(f'Duplicate customer in demand.csv: {cust}')
        demand_dict[cust] = float(row['demand'])
    if set(customers) != set(stores):
        raise ValueError('Mismatch between customers in demand.csv and stores in transportation_costs.csv')
    fixed_cost_dict = {}
    for (idx, row) in fixed_cost_df.iterrows():
        sup = normalize_id(row['Unnamed: 0'])
        if sup in fixed_cost_dict:
            raise ValueError(f'Duplicate supplier in fixed_cost.csv: {sup}')
        fixed_cost_dict[sup] = float(row['fixed_costs'])
    trans_cost_dict = {}
    for (idx, row) in trans_cost_df.iterrows():
        sup = normalize_id(row['Unnamed: 0'])
        for col in store_cols:
            store = normalize_id(col)
            val = row[col]
            if pd.isnull(val):
                raise ValueError(f'Missing transportation cost for supplier {sup}, store {store}')
            trans_cost_dict[sup, store] = float(val)
    for sup in suppliers:
        if sup not in fixed_cost_dict:
            raise ValueError(f'Missing fixed cost for supplier {sup}')
        for store in stores:
            if (sup, store) not in trans_cost_dict:
                raise ValueError(f'Missing transportation cost for supplier {sup}, store {store}')
    for store in stores:
        if store not in demand_dict:
            raise ValueError(f'Missing demand for store {store}')
    m = gp.Model('UFLP_Iowa_Liquor')
    m.Params.MIPGap = 0.0001
    y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
    x_keys = [(i, j) for i in suppliers for j in stores]
    x = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    total_fixed = gp.quicksum((fixed_cost_dict[i] * y[i] for i in suppliers))
    total_trans = gp.quicksum((trans_cost_dict[i, j] * x[i, j] for i in suppliers for j in stores))
    m.setObjective(total_fixed + total_trans, gp.GRB.MINIMIZE)
    for j in stores:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
    M = sum(demand_dict.values())
    for i in suppliers:
        for j in stores:
            m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_uflp()