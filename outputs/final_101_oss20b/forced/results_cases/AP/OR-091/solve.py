CSVQA_DATA = {'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a combination of Operations Research courses whose total credits no more than 20 and whose total interest points are maximized.', 'relationships': [], 'tables': [{'columns': ['course_id', 'credits', 'interest_points', 'discipline'], 'file_name': 'courses_42.csv', 'records': [{'data_row_number': 21, 'values': {'course_id': 'C22', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '95'}}, {'data_row_number': 22, 'values': {'course_id': 'C23', 'credits': '5', 'discipline': 'Operations Research', 'interest_points': '92'}}, {'data_row_number': 23, 'values': {'course_id': 'C24', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '86'}}, {'data_row_number': 24, 'values': {'course_id': 'C25', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '82'}}, {'data_row_number': 25, 'values': {'course_id': 'C26', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '85'}}, {'data_row_number': 26, 'values': {'course_id': 'C27', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '80'}}, {'data_row_number': 27, 'values': {'course_id': 'C28', 'credits': '4', 'discipline': 'Operations Research', 'interest_points': '88'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    table_id = 'file_0_view_0'
    table = None
    for t in CSVQA_DATA['tables']:
        if t.get('table_id') == table_id:
            table = t
            break
    if table is None:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    records = table.get('records', [])
    course_ids = []
    c = {}
    p = {}
    for rec in records:
        vals = rec.get('values', {})
        cid = vals.get('course_id')
        if cid is None:
            continue
        if vals.get('discipline') != 'Operations Research':
            continue
        credits_str = vals.get('credits')
        interest_str = vals.get('interest_points')
        if credits_str is None or interest_str is None:
            raise ValueError(f'Missing credits or interest_points for course {cid}')
        try:
            credits = int(credits_str)
            interest = int(interest_str)
        except ValueError:
            raise ValueError(f'Non-integer credits or interest_points for course {cid}')
        course_ids.append(cid)
        c[cid] = credits
        p[cid] = interest
    if not course_ids:
        raise ValueError('No Operations Research courses found')
    x = m.addVars(course_ids, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((p[i] * x[i] for i in course_ids)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((c[i] * x[i] for i in course_ids)) <= 20)
    m.optimize()
    return m
m = solve_problem()
