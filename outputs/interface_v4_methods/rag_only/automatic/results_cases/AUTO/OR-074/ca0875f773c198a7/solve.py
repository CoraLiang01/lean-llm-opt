import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    req_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv', sep=',')
    periods = list(range(len(req_df)))
    n_periods = len(periods)
    shift_length = 16
    period_to_req = {int(idx): int(req) for idx, req in req_df['Requirement'].items()}
    m = gp.Model('Waitstaff_Scheduling')
    x = m.addVars(periods, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in periods)), GRB.MINIMIZE)
    for t in periods:
        covering_starts = [s for s in periods if (t - s) % n_periods < shift_length]
        m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= period_to_req[t], name=f'cover_{t}')
    m.optimize()
    return m
m = solve_problem()