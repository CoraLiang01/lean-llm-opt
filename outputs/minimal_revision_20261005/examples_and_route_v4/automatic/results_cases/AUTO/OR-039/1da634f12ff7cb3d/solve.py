import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', sep=',')
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
warehouses_df['Warehouse ID'] = warehouses_df['Warehouse ID'].astype(str).str.strip()
products = list(products_df['ProductName'])
warehouses = list(warehouses_df['Warehouse ID'])
value = products_df.set_index('ProductName')['Value'].to_dict()
weight = products_df.set_index('ProductName')['Weight'].to_dict()
capacity = warehouses_df.set_index('Warehouse ID')['Capacity'].to_dict()
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch between product index set and value/weight parameter keys.')
if set(warehouses) != set(capacity.keys()):
    raise ValueError('Mismatch between warehouse index set and capacity parameter keys.')
keys = [(i, w) for i in products for w in warehouses]

def solve_problem(products, warehouses, value, weight, capacity, keys):
    m = gp.Model('NewCarSales_MultiKnapsack')
    x = m.addVars(keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[i] * x[i, w] for i in products for w in warehouses)), gp.GRB.MAXIMIZE)
    for w in warehouses:
        m.addConstr(gp.quicksum((weight[i] * x[i, w] for i in products)) <= capacity[w], name=f'cap_{w}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem(products, warehouses, value, weight, capacity, keys)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')