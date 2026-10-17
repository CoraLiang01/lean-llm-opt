CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, '
          'Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and '
          'an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to '
          'enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a '
          'combination of Operations Research courses whose total credits no more than 20 and whose total interest '
          'points are maximized.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['course_id', 'credits', 'interest_points', 'discipline'],
             'file_index': 0,
             'file_name': 'courses_42.csv',
             'filters': {'conditions': [{'column': 'discipline',
                                         'dtype': 'string',
                                         'evidence': 'Operations Research',
                                         'operator': 'eq',
                                         'value': 'Operations Research'}],
                         'logic': 'and'},
             'original_rows': 42,
             'records': [{'source_row': 21,
                          'values': {'course_id': 'C22',
                                     'credits': '5',
                                     'discipline': 'Operations Research',
                                     'interest_points': '95'}},
                         {'source_row': 22,
                          'values': {'course_id': 'C23',
                                     'credits': '5',
                                     'discipline': 'Operations Research',
                                     'interest_points': '92'}},
                         {'source_row': 23,
                          'values': {'course_id': 'C24',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '86'}},
                         {'source_row': 24,
                          'values': {'course_id': 'C25',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '82'}},
                         {'source_row': 25,
                          'values': {'course_id': 'C26',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '85'}},
                         {'source_row': 26,
                          'values': {'course_id': 'C27',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '80'}},
                         {'source_row': 27,
                          'values': {'course_id': 'C28',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '88'}}],
             'returned_rows': 7,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == 'file_0_view_0':
            target_table = tbl
            break
    if target_table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    credits = {}
    interest_points = {}
    course_ids = []
    for rec in records:
        vals = rec.get('values', {})
        cid = vals.get('course_id')
        if cid is None:
            raise ValueError(f'Missing course_id in record {rec}')
        discipline = vals.get('discipline')
        if discipline != 'Operations Research':
            continue
        cr = vals.get('credits')
        ip = vals.get('interest_points')
        if cr is None or ip is None:
            raise ValueError(f'Missing credits or interest_points for course {cid}')
        try:
            cr_int = int(cr)
            ip_int = int(ip)
        except ValueError:
            raise ValueError(f'Non-integer credits or interest_points for course {cid}')
        credits[cid] = cr_int
        interest_points[cid] = ip_int
        course_ids.append(cid)
    for cid in course_ids:
        if cid not in credits or cid not in interest_points:
            raise ValueError(f'Missing parameters for course {cid}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(course_ids, lb=0, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((interest_points[cid] * x[cid] for cid in course_ids)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((credits[cid] * x[cid] for cid in course_ids)) <= 20, name='credit_limit')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()