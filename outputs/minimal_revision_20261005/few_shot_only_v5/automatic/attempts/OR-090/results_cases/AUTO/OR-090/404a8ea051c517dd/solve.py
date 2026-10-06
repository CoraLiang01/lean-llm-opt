import gurobipy as gp
import pandas as pd
import numpy as np
import math
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv'
resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv'

def solve_problem():
    df_prod = pd.read_csv(products_path, sep=',')
    df_res = pd.read_csv(resources_path, sep=',')
    required_prod_cols = ['product', 'profit_per_unit', 'r1_per_unit', 'r2_per_unit', 'r3_per_unit', 'upper_demand_units', 'batch_size_units']
    for col in required_prod_cols:
        if col not in df_prod.columns:
            raise KeyError(f"Missing required column '{col}' in factory_products_100.csv")
    required_res_cols = ['resource', 'capacity']
    for col in required_res_cols:
        if col not in df_res.columns:
            raise KeyError(f"Missing required column '{col}' in resources_capacities.csv")
    products = df_prod['product'].astype(str).tolist()
    if len(products) != 100:
        raise ValueError(f'Expected 100 products, found {len(products)} in factory_products_100.csv')
    resources = df_res['resource'].astype(str).tolist()
    expected_resources = {'R1', 'R2', 'R3'}
    if set(resources) != expected_resources:
        raise ValueError(f'Expected resources {expected_resources}, found {set(resources)} in resources_capacities.csv')
    batch_sizes = df_prod.set_index('product')['batch_size_units'].to_dict()
    unique_batch_sizes = set(batch_sizes.values())
    if len(unique_batch_sizes) != 1 or 10 not in unique_batch_sizes:
        raise ValueError(f'Batch size must be 10 for all products, found {unique_batch_sizes}')
    batch_size = 10
    profit_per_unit = df_prod.set_index('product')['profit_per_unit'].to_dict()
    r1_per_unit = df_prod.set_index('product')['r1_per_unit'].to_dict()
    r2_per_unit = df_prod.set_index('product')['r2_per_unit'].to_dict()
    r3_per_unit = df_prod.set_index('product')['r3_per_unit'].to_dict()
    upper_demand_units = df_prod.set_index('product')['upper_demand_units'].to_dict()
    capacity = df_res.set_index('resource')['capacity'].to_dict()
    for r in expected_resources:
        if r not in capacity:
            raise KeyError(f"Resource '{r}' not found in resources_capacities.csv")
    batch_upper_bound = {}
    for p in products:
        ud = upper_demand_units[p]
        if not (isinstance(ud, (int, float)) and ud >= 0):
            raise ValueError(f'upper_demand_units for product {p} is invalid: {ud}')
        batch_upper_bound[p] = int(math.floor(ud / batch_size))
    m = gp.Model('BatchProduction')
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, ub=[batch_upper_bound[p] for p in products], name='')
    m.setObjective(gp.quicksum((batch_size * x[p] * profit_per_unit[p] for p in products)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x[p] * batch_size * r1_per_unit[p] for p in products)) <= capacity['R1'], name='r1')
    m.addConstr(gp.quicksum((x[p] * batch_size * r2_per_unit[p] for p in products)) <= capacity['R2'], name='r2')
    m.addConstr(gp.quicksum((x[p] * batch_size * r3_per_unit[p] for p in products)) <= capacity['R3'], name='r3')
    m.addConstrs((x[p] * batch_size <= upper_demand_units[p] for p in products), name='')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')