import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
    df = pd.read_csv(path, sep=',')
    shifts = df['Shift'].astype(int).tolist()
    n_periods = len(shifts)
    if n_periods != 24:
        raise ValueError('Expected 24 periods, got {}'.format(n_periods))
    required = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))
    m = gp.Model('bus_staff_scheduling')
    x = m.addVars(shifts, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[t] for t in shifts)), GRB.MINIMIZE)
    for s in shifts:
        cover_starts = [(s - i - 1) % n_periods + 1 for i in range(4)]
        m.addConstr(gp.quicksum((x[t] for t in cover_starts)) >= required[s], name=f'cover_{s}')
    m.optimize()
    return m
m = solve_problem()