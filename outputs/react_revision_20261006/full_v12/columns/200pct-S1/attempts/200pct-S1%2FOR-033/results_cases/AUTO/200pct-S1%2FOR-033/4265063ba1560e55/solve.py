CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A certain restaurant operates 24 hours a day and requires a minimum number of waitstaff as shown in '
          '44.csv.\n'
          '\n'
          '    Each waitstaff works continuously for 8 hours a day. The goal is to find the minimum number of '
          'waitstaff that satisfies the above conditions and represent this problem as a linear programming model.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Time', 'Requirement'],
             'file_index': 0,
             'file_name': '44.csv',
             'filters': {},
             'original_rows': 48,
             'records': [{'source_row': 0, 'values': {'Requirement': '2', 'Time': '2:00am - 2:30am'}},
                         {'source_row': 1, 'values': {'Requirement': '3', 'Time': '2:30am - 3:00am'}},
                         {'source_row': 2, 'values': {'Requirement': '4', 'Time': '3:00am - 3:30am'}},
                         {'source_row': 3, 'values': {'Requirement': '6', 'Time': '3:30am - 4:00am'}},
                         {'source_row': 4, 'values': {'Requirement': '5', 'Time': '4:00am - 4:30am'}},
                         {'source_row': 5, 'values': {'Requirement': '4', 'Time': '4:30am - 5:00am'}},
                         {'source_row': 6, 'values': {'Requirement': '5', 'Time': '5:00am - 5:30am'}},
                         {'source_row': 7, 'values': {'Requirement': '6', 'Time': '5:30am - 6:00am'}},
                         {'source_row': 8, 'values': {'Requirement': '7', 'Time': '6:00am - 6:30am'}},
                         {'source_row': 9, 'values': {'Requirement': '8', 'Time': '6:30am - 7:00am'}},
                         {'source_row': 10, 'values': {'Requirement': '9', 'Time': '7:00am - 7:30am'}},
                         {'source_row': 11, 'values': {'Requirement': '9', 'Time': '7:30am - 8:00am'}},
                         {'source_row': 12, 'values': {'Requirement': '8', 'Time': '8:00am - 8:30am'}},
                         {'source_row': 13, 'values': {'Requirement': '8', 'Time': '8:30am - 9:00am'}},
                         {'source_row': 14, 'values': {'Requirement': '9', 'Time': '9:00am - 9:30am'}},
                         {'source_row': 15, 'values': {'Requirement': '9', 'Time': '9:30am - 10:00am'}},
                         {'source_row': 16, 'values': {'Requirement': '10', 'Time': '10:00am - 10:30am'}},
                         {'source_row': 17, 'values': {'Requirement': '12', 'Time': '10:30am - 11:00am'}},
                         {'source_row': 18, 'values': {'Requirement': '11', 'Time': '11:00am - 11:30am'}},
                         {'source_row': 19, 'values': {'Requirement': '11', 'Time': '11:30am - 12:00pm'}},
                         {'source_row': 20, 'values': {'Requirement': '12', 'Time': '12:00pm - 12:30pm'}},
                         {'source_row': 21, 'values': {'Requirement': '11', 'Time': '12:30pm - 1:00pm'}},
                         {'source_row': 22, 'values': {'Requirement': '10', 'Time': '1:00pm - 1:30pm'}},
                         {'source_row': 23, 'values': {'Requirement': '9', 'Time': '1:30pm - 2:00pm'}},
                         {'source_row': 24, 'values': {'Requirement': '8', 'Time': '2:00pm - 2:30pm'}},
                         {'source_row': 25, 'values': {'Requirement': '7', 'Time': '2:30pm - 3:00pm'}},
                         {'source_row': 26, 'values': {'Requirement': '6', 'Time': '3:00pm - 3:30pm'}},
                         {'source_row': 27, 'values': {'Requirement': '5', 'Time': '3:30pm - 4:00pm'}},
                         {'source_row': 28, 'values': {'Requirement': '5', 'Time': '4:00pm - 4:30pm'}},
                         {'source_row': 29, 'values': {'Requirement': '6', 'Time': '4:30pm - 5:00pm'}},
                         {'source_row': 30, 'values': {'Requirement': '7', 'Time': '5:00pm - 5:30pm'}},
                         {'source_row': 31, 'values': {'Requirement': '8', 'Time': '5:30pm - 6:00pm'}},
                         {'source_row': 32, 'values': {'Requirement': '9', 'Time': '6:00pm - 6:30pm'}},
                         {'source_row': 33, 'values': {'Requirement': '10', 'Time': '6:30pm - 7:00pm'}},
                         {'source_row': 34, 'values': {'Requirement': '9', 'Time': '7:00pm - 7:30pm'}},
                         {'source_row': 35, 'values': {'Requirement': '8', 'Time': '7:30pm - 8:00pm'}},
                         {'source_row': 36, 'values': {'Requirement': '7', 'Time': '8:00pm - 8:30pm'}},
                         {'source_row': 37, 'values': {'Requirement': '6', 'Time': '8:30pm - 9:00pm'}},
                         {'source_row': 38, 'values': {'Requirement': '5', 'Time': '9:00pm - 9:30pm'}},
                         {'source_row': 39, 'values': {'Requirement': '4', 'Time': '9:30pm - 10:00pm'}},
                         {'source_row': 40, 'values': {'Requirement': '4', 'Time': '10:00pm - 10:30pm'}},
                         {'source_row': 41, 'values': {'Requirement': '3', 'Time': '10:30pm - 11:00pm'}},
                         {'source_row': 42, 'values': {'Requirement': '3', 'Time': '11:00pm - 11:30pm'}},
                         {'source_row': 43, 'values': {'Requirement': '3', 'Time': '11:30pm - 12:00am'}},
                         {'source_row': 44, 'values': {'Requirement': '3', 'Time': '12:00am - 12:30am'}},
                         {'source_row': 45, 'values': {'Requirement': '4', 'Time': '12:30am - 1:00am'}},
                         {'source_row': 46, 'values': {'Requirement': '4', 'Time': '1:00am - 1:30am'}},
                         {'source_row': 47, 'values': {'Requirement': '4', 'Time': '1:30am - 2:00am'}}],
             'returned_rows': 48,
             'role': 'waitstaff requirement by time period',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    T = list(frame.index)
    S = list(frame.index)
    time_labels = frame['Time'].tolist()
    r_t = {}
    for (t, row) in frame.iterrows():
        try:
            r_t[t] = float(row['Requirement'])
        except Exception:
            raise ValueError(f"Requirement missing or invalid at row {t}: {row['Requirement']}")
    if len(r_t) != 48:
        raise ValueError('Missing requirement data for some intervals.')
    m = gp.Model('WaitstaffScheduling')
    x_vars = m.addVars(S, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x_vars[s] for s in S)), gp.GRB.MINIMIZE)
    for t in T:
        covered_s = []
        for s in S:
            covered = [(s + offset) % 48 for offset in range(16)]
            if t in covered:
                covered_s.append(s)
        if not covered_s:
            raise ValueError(f'No shift covers interval {t}')
        m.addConstr(gp.quicksum((x_vars[s] for s in covered_s)) >= r_t[t], name=f'cover_{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)