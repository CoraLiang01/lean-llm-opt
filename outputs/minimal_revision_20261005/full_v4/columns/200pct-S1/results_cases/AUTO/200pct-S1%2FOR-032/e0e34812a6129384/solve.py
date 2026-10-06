import gurobipy as gp
import pandas as pd
import numpy as np
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',')
discipline_mask = df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses_df = df[discipline_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = list(or_courses_df['course_id'])
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row['course_id']
    credits[cid] = int(row['credits'])
    interest_points[cid] = int(row['interest_points'])
if set(credits.keys()) != set(course_ids) or set(interest_points.keys()) != set(course_ids):
    raise ValueError('Missing credits or interest_points data for some Operations Research courses.')

def solve_or_knapsack(course_ids, credits, interest_points, credit_limit=20):
    m = gp.Model('OR_Course_Selection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[c] * x[c] for c in course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[c] * x[c] for c in course_ids)) <= credit_limit, name='credit_limit')
    m.optimize()
    return m
m = solve_or_knapsack(course_ids, credits, interest_points, credit_limit=20)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')