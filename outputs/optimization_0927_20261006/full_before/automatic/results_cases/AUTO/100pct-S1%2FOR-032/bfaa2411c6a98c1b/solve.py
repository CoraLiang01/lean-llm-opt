import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
ops_research_df = df[ops_research_mask].copy()
ops_course_ids = ops_research_df['course_id'].astype(str).tolist()
credits = {}
interest_points = {}
for (_, row) in ops_research_df.iterrows():
    cid = str(row['course_id'])
    credits[cid] = int(row['credits'])
    interest_points[cid] = int(row['interest_points'])
if len(ops_course_ids) == 0:
    raise ValueError('No Operations Research courses found in the data.')
for cid in ops_course_ids:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for course_id {cid}.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(ops_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in ops_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in ops_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in ops_course_ids if x[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits} / 20')
    print('\nSelected Operations Research courses:')
    for cid in selected:
        cname = ops_research_df.loc[ops_research_df['course_id'] == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} (Credits: {credits[cid]}, Interest Points: {interest_points[cid]})')
else:
    print(f'No optimal solution found. Status: {m.status}')