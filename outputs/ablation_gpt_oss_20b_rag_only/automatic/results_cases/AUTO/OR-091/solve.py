CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, '
          'Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and '
          'an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to '
          'enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a '
          'combination of Operations Research courses whose total credits no more than 20 and whose total interest '
          'points are maximized.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['course_id', 'discipline', 'credits', 'interest_points'],
             'file_index': 0,
             'file_name': 'courses_42.csv',
             'filters': {'conditions': [{'column': 'discipline',
                                         'dtype': 'string',
                                         'evidence': 'Operations Research',
                                         'inclusive': 'both',
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
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA.get('tables', [])
    table = next((t for t in tables if t.get('table_id') == 'file_0_view_0'), None)
    if table is None:
        raise ValueError('Required table file_0_view_0 not found')
    records = table.get('records', [])
    I = []
    c = {}
    p = {}
    for rec in records:
        vals = rec.get('values', {})
        course_id = vals.get('course_id')
        if course_id is None:
            raise ValueError(f"Missing course_id in record {rec.get('source_row')}")
        credits_str = vals.get('credits')
        interest_str = vals.get('interest_points')
        if credits_str is None or interest_str is None:
            raise ValueError(f"Missing credits or interest_points in record {rec.get('source_row')}")
        try:
            credits = float(credits_str)
            interest = float(interest_str)
        except ValueError:
            raise ValueError(f"Non-numeric credits or interest_points in record {rec.get('source_row')}")
        I.append(course_id)
        c[course_id] = credits
        p[course_id] = interest
    if not I:
        raise ValueError('No Operations Research courses found')
    if set(c.keys()) != set(I):
        raise ValueError('Mismatch between courses and credits')
    if set(p.keys()) != set(I):
        raise ValueError('Mismatch between courses and interest points')
    m = gp.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((p[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((c[i] * x[i] for i in I)) <= 20, name='CreditLimit')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()