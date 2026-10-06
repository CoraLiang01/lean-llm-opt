import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_mask = courses_df['discipline'].astype(str).str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError('No Operations Research courses found in the input data.')
or_course_ids = list(or_courses_df['course_id'])
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row['course_id']
    if pd.isnull(row['credits']) or pd.isnull(row['interest_points']):
        raise ValueError(f'Missing credits or interest_points for course_id {cid}')
    credits[cid] = float(row['credits'])
    interest_points[cid] = float(row['interest_points'])
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')

def solve_or_course_selection(or_course_ids, credits, interest_points):
    m = gp.Model('OR_Course_Selection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[i] * x[i] for i in or_course_ids)) <= 20, name='credit_limit')
    m.optimize()
    return m
m = solve_or_course_selection(or_course_ids, credits, interest_points)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')