import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_mask = courses_df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses = courses_df.loc[or_mask].copy()
if or_courses.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = list(or_courses['course_id'])
credits = dict(zip(or_courses['course_id'], or_courses['credits']))
interest_points = dict(zip(or_courses['course_id'], or_courses['interest_points']))
if set(credits.keys()) != set(course_ids) or set(interest_points.keys()) != set(course_ids):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')

def solve_or_knapsack(course_ids, credits, interest_points):
    m = gp.Model('OR_Knapsack')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[i] * x[i] for i in course_ids)) <= 20, name='credit_limit')
    m.optimize()
    return m
m = solve_or_knapsack(course_ids, credits, interest_points)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')