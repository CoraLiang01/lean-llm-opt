import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv', sep=',')
resources_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv', sep=',')
product_ids = products_df['product'].tolist()
batch_size = int(products_df['batch_size_units'].iloc[0])
profit_per_unit = dict(zip(products_df['product'], products_df['profit_per_unit']))
r1_per_unit = dict(zip(products_df['product'], products_df['r1_per_unit']))
r2_per_unit = dict(zip(products_df['product'], products_df['r2_per_unit']))
r3_per_unit = dict(zip(products_df['product'], products_df['r3_per_unit']))
upper_demand_units = dict(zip(products_df['product'], products_df['upper_demand_units']))
capacity = dict(zip(resources_df['resource'], resources_df['capacity']))
m = gp.Model('BatchProduction')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER)
m.addConstrs((batch_size * x[i] <= upper_demand_units[i] for i in product_ids))
m.addConstr(gp.quicksum((batch_size * r1_per_unit[i] * x[i] for i in product_ids)) <= capacity['R1'])
m.addConstr(gp.quicksum((batch_size * r2_per_unit[i] * x[i] for i in product_ids)) <= capacity['R2'])
m.addConstr(gp.quicksum((batch_size * r3_per_unit[i] * x[i] for i in product_ids)) <= capacity['R3'])
m.setObjective(gp.quicksum((batch_size * profit_per_unit[i] * x[i] for i in product_ids)), sense=gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('\nProduction plan (batches and units):')
    for i in product_ids:
        batches = x[i].X
        if batches > 0:
            units = batches * batch_size
            print(f'  {i}: {int(batches)} batches ({int(units)} units)')
else:
    print(f'Optimization ended with status {m.status}')