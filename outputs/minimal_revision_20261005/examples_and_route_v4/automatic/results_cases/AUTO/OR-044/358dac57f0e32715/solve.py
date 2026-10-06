import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype={'SectionID': np.int64, 'Capacity': np.int64})
products_df = pd.read_csv(products_path, sep=',', dtype={'ProductName': np.int64, 'Value': np.int64, 'Weight': np.int64})
sections = capacity_df['SectionID'].unique()
products = products_df['ProductName'].unique()
capacity = dict(zip(capacity_df['SectionID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(sections) != set(capacity.keys()):
    raise ValueError('Mismatch between section index set and capacity keys.')
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch between product index set and value/weight keys.')
index_pairs = [(i, j) for i in sections for j in products]

def solve_problem():
    m = gp.Model('SupermarketProductSelection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(index_pairs, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[j] * x[i, j] for (i, j) in index_pairs)), gp.GRB.MAXIMIZE)
    for i in sections:
        m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i], name=f'cap_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')