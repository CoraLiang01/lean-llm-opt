import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_mask = courses_df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
OR_COURSES = list(or_courses_df['course_id'])
credits = dict(zip(or_courses_df['course_id'], or_courses_df['credits']))
interest_points = dict(zip(or_courses_df['course_id'], or_courses_df['interest_points']))
if set(credits.keys()) != set(OR_COURSES) or set(interest_points.keys()) != set(OR_COURSES):
    raise ValueError('Missing credits or interest_points data for some Operations Research courses.')

def solve_or_knapsack(OR_COURSES, credits, interest_points):
    m = gp.Model('OR_Course_Selection')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(OR_COURSES, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in OR_COURSES)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[i] * x[i] for i in OR_COURSES)) <= 20, name='credit_limit')
    m.optimize()
    return m
m = solve_or_knapsack(OR_COURSES, credits, interest_points)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')