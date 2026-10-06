import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv'
resources_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv'
products_df = pd.read_csv(products_path, sep=',')
resources_df = pd.read_csv(resources_path, sep=',')
required_product_cols = ['product', 'profit_per_unit', 'r1_per_unit', 'r2_per_unit', 'r3_per_unit', 'upper_demand_units', 'batch_size_units']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Missing required column '{col}' in factory_products_100.csv")
required_resource_cols = ['resource', 'capacity']
for col in required_resource_cols:
    if col not in resources_df.columns:
        raise KeyError(f"Missing required column '{col}' in resources_capacities.csv")
product_ids = products_df['product'].astype(str).tolist()
resource_ids = resources_df['resource'].astype(str).tolist()
batch_sizes = products_df['batch_size_units'].unique()
if len(batch_sizes) != 1 or batch_sizes[0] != 10:
    raise ValueError(f'batch_size_units must be constant and equal to 10 for all products, got {batch_sizes}')
batch_size = 10
profit_per_unit = products_df.set_index('product')['profit_per_unit'].astype(float).to_dict()
r1_per_unit = products_df.set_index('product')['r1_per_unit'].astype(float).to_dict()
r2_per_unit = products_df.set_index('product')['r2_per_unit'].astype(float).to_dict()
r3_per_unit = products_df.set_index('product')['r3_per_unit'].astype(float).to_dict()
upper_demand_units = products_df.set_index('product')['upper_demand_units'].astype(int).to_dict()
resource_capacities = resources_df.set_index('resource')['capacity'].astype(float).to_dict()
resource_per_unit = {'R1': r1_per_unit, 'R2': r2_per_unit, 'R3': r3_per_unit}
for r in ['R1', 'R2', 'R3']:
    if r not in resource_capacities:
        raise KeyError(f"Resource '{r}' not found in resources_capacities.csv")
    if r not in resource_per_unit:
        raise KeyError(f"Resource per-unit consumption for '{r}' not found in products data")
for pid in product_ids:
    for d in [profit_per_unit, r1_per_unit, r2_per_unit, r3_per_unit, upper_demand_units]:
        if pid not in d:
            raise KeyError(f"Product '{pid}' missing required data in one of the parameter columns")

def solve_problem():
    m = gp.Model('BatchProductionPlanning')
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((batch_size * x[pid] * profit_per_unit[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
    for r in ['R1', 'R2', 'R3']:
        m.addConstr(gp.quicksum((batch_size * x[pid] * resource_per_unit[r][pid] for pid in product_ids)) <= resource_capacities[r], name=f'res_{r}')
    for pid in product_ids:
        m.addConstr(batch_size * x[pid] <= upper_demand_units[pid], name=f'demand_{pid}')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')