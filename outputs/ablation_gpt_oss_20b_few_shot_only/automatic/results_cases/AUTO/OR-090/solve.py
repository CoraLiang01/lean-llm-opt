import pandas as pd
import gurobipy
from gurobipy import GRB
PRODUCTS_CSV = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv'
RESOURCES_CSV = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv'

def read_products():
    df = pd.read_csv(PRODUCTS_CSV, dtype=str, keep_default_na=False)
    required_cols = ['product', 'profit_per_unit', 'r1_per_unit', 'r2_per_unit', 'r3_per_unit', 'upper_demand_units', 'batch_size_units']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in products CSV: {missing}')
    df['profit_per_unit'] = df['profit_per_unit'].astype(float)
    df['r1_per_unit'] = df['r1_per_unit'].astype(float)
    df['r2_per_unit'] = df['r2_per_unit'].astype(float)
    df['r3_per_unit'] = df['r3_per_unit'].astype(float)
    df['upper_demand_units'] = df['upper_demand_units'].astype(int)
    df['batch_size_units'] = df['batch_size_units'].astype(int)
    products = df['product'].tolist()
    profit = dict(zip(products, df['profit_per_unit']))
    r1 = dict(zip(products, df['r1_per_unit']))
    r2 = dict(zip(products, df['r2_per_unit']))
    r3 = dict(zip(products, df['r3_per_unit']))
    demand = dict(zip(products, df['upper_demand_units']))
    batch = dict(zip(products, df['batch_size_units']))
    if not all((b == 10 for b in batch.values())):
        raise ValueError('Batch size is not 10 for all products')
    return (products, profit, r1, r2, r3, demand)

def read_resources():
    df = pd.read_csv(RESOURCES_CSV, dtype=str, keep_default_na=False)
    if 'resource' not in df.columns or 'capacity' not in df.columns:
        raise ValueError("Resources CSV must contain 'resource' and 'capacity' columns")
    df['capacity'] = df['capacity'].astype(float)
    capacities = dict(zip(df['resource'], df['capacity']))
    for r in ['R1', 'R2', 'R3']:
        if r not in capacities:
            raise ValueError(f'Missing capacity for resource {r}')
    return (capacities['R1'], capacities['R2'], capacities['R3'])

def solve_problem():
    (products, profit, r1, r2, r3, demand) = read_products()
    (c_R1, c_R2, c_R3) = read_resources()
    m = gurobipy.Model('FactoryBatchProduction')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((10 * profit[i] * x[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((10 * r1[i] * x[i] for i in products)) <= c_R1, 'R1_capacity'), (gurobipy.quicksum((10 * r2[i] * x[i] for i in products)) <= c_R2, 'R2_capacity'), (gurobipy.quicksum((10 * r3[i] * x[i] for i in products)) <= c_R3, 'R3_capacity'), name='')
    m.addConstrs((10 * x[i] <= demand[i] for i in products), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()