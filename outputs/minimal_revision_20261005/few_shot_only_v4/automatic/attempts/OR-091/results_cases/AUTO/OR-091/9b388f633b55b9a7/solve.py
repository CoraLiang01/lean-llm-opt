import gurobipy as gp
import pandas as pd
import numpy as np
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',')
or_mask = courses_df['discipline'].astype(str).str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found with discipline 'Operations Research'.")
or_course_ids = list(or_courses_df['course_id'])
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row['course_id']
    try:
        credits[cid] = float(row['credits'])
        interest_points[cid] = float(row['interest_points'])
    except Exception as e:
        raise ValueError(f'Invalid or missing data for course_id {cid}: {e}')
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in parameter keys for Operations Research courses.')

def solve_or_knapsack(or_course_ids, credits, interest_points, credit_limit=20):
    m = gp.Model('OR_Course_Selection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in or_course_ids)) <= credit_limit, name='credit_limit')
    m.optimize()
    return m
m = solve_or_knapsack(or_course_ids, credits, interest_points, credit_limit=20)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')