import pandas as pd
import gurobipy as gp
from gurobipy import GRB
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_courses_df = courses_df[courses_df['discipline'].str.casefold().str.strip() == 'operations research']
or_course_ids = list(or_courses_df['course_id'])
if len(or_course_ids) == 0:
    raise ValueError('No Operations Research courses found in the data.')
credits = {}
interest_points = {}
for (_, row) in or_courses_df.iterrows():
    cid = str(row['course_id'])
    if pd.isnull(row['credits']) or pd.isnull(row['interest_points']):
        raise ValueError(f'Missing data for course_id {cid}')
    credits[cid] = int(row['credits'])
    interest_points[cid] = int(row['interest_points'])
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in parameter coverage for Operations Research courses.')
m = gp.Model('or_course_selection')
x = m.addVars(or_course_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in or_course_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in or_course_ids:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.Status}')