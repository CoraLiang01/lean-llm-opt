import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', dtype=str, keep_default_na=False)
    or_mask = df['discipline'].str.casefold() == 'operations research'
    or_courses_df = df[or_mask].copy()
    if or_courses_df.empty:
        raise ValueError('No Operations Research courses found in the input data.')
    or_course_ids = list(or_courses_df['course_id'])
    try:
        credits = {}
        interest_points = {}
        for (idx, row) in or_courses_df.iterrows():
            cid = row['course_id']
            if cid in credits:
                raise ValueError(f'Duplicate course_id found: {cid}')
            try:
                credits[cid] = float(row['credits'])
            except Exception:
                raise ValueError(f"Invalid or missing credits for course_id {cid}: {row['credits']}")
            try:
                interest_points[cid] = float(row['interest_points'])
            except Exception:
                raise ValueError(f"Invalid or missing interest_points for course_id {cid}: {row['interest_points']}")
    except Exception as e:
        raise
    if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
        raise ValueError('Mismatch in parameter keys for OR courses.')
    m = gp.Model('OR_Course_Selection')
    x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for cid in or_course_ids:
            print(f'{x_vars[cid].VarName} {x_vars[cid].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()