import gurobipy as gp
import pandas as pd
import numpy as np
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',')
discipline_mask = df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses_df = df[discipline_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found with discipline 'Operations Research'.")
course_ids = list(or_courses_df['course_id'])
credits = dict(zip(or_courses_df['course_id'], or_courses_df['credits']))
interest_points = dict(zip(or_courses_df['course_id'], or_courses_df['interest_points']))
if set(course_ids) != set(credits.keys()) or set(course_ids) != set(interest_points.keys()):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')

def solve_or_knapsack(course_ids, credits, interest_points, credit_limit):
    m = gp.Model('OR_Course_Selection')
    x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[i] * x[i] for i in course_ids)) <= credit_limit, name='credit_limit')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_or_knapsack(course_ids, credits, interest_points, credit_limit=20)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')