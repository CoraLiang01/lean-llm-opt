import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
ops_df = df[ops_research_mask].copy()
I = list(ops_df['course_id'])
credits = {}
interest_points = {}
for idx, row in ops_df.iterrows():
    cid = row['course_id']
    credits[cid] = int(row['credits'])
    interest_points[cid] = int(row['interest_points'])
if len(I) == 0:
    raise ValueError('No Operations Research courses found in the data.')
if set(credits.keys()) != set(I) or set(interest_points.keys()) != set(I):
    raise ValueError('Missing credits or interest_points for some Operations Research courses.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(I, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in I)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in I)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in I if x[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    total_interest = sum((interest_points[i] for i in selected))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits} / 20')
    print('\nSelected Operations Research courses:')
    for i in selected:
        cname = ops_df.loc[ops_df['course_id'] == i, 'course_name'].values[0]
        print(f'  {i}: {cname} (Credits: {credits[i]}, Interest Points: {interest_points[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')