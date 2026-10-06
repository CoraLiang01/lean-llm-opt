import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
discipline_mask = courses_df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses_df = courses_df[discipline_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
C = list(or_courses_df['course_id'])
interest_points = dict(zip(or_courses_df['course_id'], or_courses_df['interest_points']))
credits = dict(zip(or_courses_df['course_id'], or_courses_df['credits']))
if set(interest_points.keys()) != set(C) or set(credits.keys()) != set(C):
    raise ValueError('Mismatch in course_id keys for interest_points or credits.')

def solve_problem():
    m = gp.Model('OR_Course_Selection')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(C, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[c] * x[c] for c in C)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[c] * x[c] for c in C)) <= 20, name='credit_limit')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')