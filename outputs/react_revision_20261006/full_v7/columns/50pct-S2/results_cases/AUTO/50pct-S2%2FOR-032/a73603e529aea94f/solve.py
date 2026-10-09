import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = courses_df[discipline_col].str.casefold().str.strip() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = list(or_courses_df['course_id'])
try:
    credits = or_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
    interest_points = or_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'credits' or 'interest_points' to int: {e}")
if set(credits.keys()) != set(course_ids) or set(interest_points.keys()) != set(course_ids):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')
CREDIT_LIMIT = 20

def solve_or_knapsack(course_ids, credits, interest_points, credit_limit):
    m = gp.Model('OR_Course_Selection')
    x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in course_ids)) <= credit_limit, name='credit_limit')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_or_knapsack(course_ids, credits, interest_points, CREDIT_LIMIT)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')