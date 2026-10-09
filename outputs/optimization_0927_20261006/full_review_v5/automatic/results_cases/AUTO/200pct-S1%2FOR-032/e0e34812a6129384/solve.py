import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = courses_df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = or_courses_df['course_id'].tolist()

def extract_int_column(df, col, ids):
    vals = {}
    for (idx, row) in df.iterrows():
        cid = row['course_id']
        try:
            val = int(row[col])
        except Exception as e:
            raise ValueError(f"Invalid integer in column '{col}' for course_id '{cid}': {row[col]}")
        vals[cid] = val
    missing = set(ids) - set(vals.keys())
    if missing:
        raise ValueError(f'Missing {col} values for course_ids: {missing}')
    return vals
credits = extract_int_column(or_courses_df, 'credits', course_ids)
interest_points = extract_int_column(or_courses_df, 'interest_points', course_ids)
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[cid] * x_vars[cid].X for cid in course_ids))
    total_credits = sum((credits[cid] * x_vars[cid].X for cid in course_ids))
    print(f'Optimal total interest points: {total_interest:.0f}')
    print(f'Total credits used: {total_credits:.0f} / 20')
    print('\nSelected Operations Research courses:')
    for (idx, cid) in enumerate(course_ids):
        if x_vars[cid].X > 0.5:
            row = or_courses_df[or_courses_df['course_id'] == cid].iloc[0]
            print(f"  {row['course_id']}: {row['course_name']} (Credits: {credits[cid]}, Interest: {interest_points[cid]})")
else:
    print(f'No optimal solution found. Status: {m.status}')