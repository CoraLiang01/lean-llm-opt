import pandas as pd
import gurobipy as gp
from gurobipy import GRB
value_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv', sep=',')
value_df['item'] = value_df['item'].astype(int)
value_df['value'] = value_df['value'].astype(int)
value_df['weight'] = value_df['weight'].astype(int)
items = value_df['item'].tolist()
value_dict = dict(zip(value_df['item'], value_df['value']))
weight_dict = dict(zip(value_df['item'], value_df['weight']))

def solve_problem():
    m = gp.Model('shopping_centre_knapsack')
    x = m.addVars(items, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
    m.optimize()
    return m
m = solve_problem()