CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Assign six construction projects to the eight listed managers. Each project needs exactly one person, and '
          'each person may take at most one project. Exclude anyone with on_leave=1. The assigned person must meet '
          'required_skill, with Junior < Intermediate < Senior < Expert. In the cost matrix, project IDs are columns '
          'and worker IDs identify rows. A blank cell forbids that assignment. Minimize the sum of assignment costs, '
          'and report the minimum in USD cents.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['worker_id', 'skill', 'on_leave'],
             'file_index': 0,
             'file_name': 'export_01.csv',
             'filters': {'conditions': [{'column': 'on_leave',
                                         'dtype': 'number',
                                         'evidence': 'on_leave=1',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'eq',
                                         'value': '0'}],
                         'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W12'}},
                         {'source_row': 1, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W06'}},
                         {'source_row': 2, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W04'}},
                         {'source_row': 4, 'values': {'on_leave': '0', 'skill': 'Junior', 'worker_id': 'W11'}},
                         {'source_row': 5, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W10'}},
                         {'source_row': 6, 'values': {'on_leave': '0', 'skill': 'Intermediate', 'worker_id': 'W02'}},
                         {'source_row': 7, 'values': {'on_leave': '0', 'skill': 'Expert', 'worker_id': 'W00'}}],
             'returned_rows': 7,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['project_id', 'required_skill'],
             'file_index': 1,
             'file_name': 'export_02.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'project_id': 'P00', 'required_skill': 'Junior'}},
                         {'source_row': 1, 'values': {'project_id': 'P01', 'required_skill': 'Junior'}},
                         {'source_row': 2, 'values': {'project_id': 'P02', 'required_skill': 'Junior'}},
                         {'source_row': 3, 'values': {'project_id': 'P03', 'required_skill': 'Junior'}},
                         {'source_row': 4, 'values': {'project_id': 'P04', 'required_skill': 'Junior'}},
                         {'source_row': 5, 'values': {'project_id': 'P05', 'required_skill': 'Intermediate'}}],
             'returned_rows': 6,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'},
            {'columns': ['worker_id', 'P00', 'P01', 'P02', 'P03', 'P04', 'P05'],
             'file_index': 2,
             'file_name': 'export_03.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'P00': '1091',
                                     'P01': '324',
                                     'P02': '379',
                                     'P03': '272',
                                     'P04': '',
                                     'P05': '',
                                     'worker_id': 'W11'}},
                         {'source_row': 1,
                          'values': {'P00': '1176',
                                     'P01': '1111',
                                     'P02': '1380',
                                     'P03': '542',
                                     'P04': '158',
                                     'P05': '922',
                                     'worker_id': 'W10'}},
                         {'source_row': 2,
                          'values': {'P00': '',
                                     'P01': '822',
                                     'P02': '',
                                     'P03': '223',
                                     'P04': '1155',
                                     'P05': '1055',
                                     'worker_id': 'W06'}},
                         {'source_row': 3,
                          'values': {'P00': '5',
                                     'P01': '',
                                     'P02': '20',
                                     'P03': '',
                                     'P04': '18',
                                     'P05': '7',
                                     'worker_id': 'W01'}},
                         {'source_row': 4,
                          'values': {'P00': '',
                                     'P01': '1397',
                                     'P02': '953',
                                     'P03': '714',
                                     'P04': '205',
                                     'P05': '',
                                     'worker_id': 'W02'}},
                         {'source_row': 5,
                          'values': {'P00': '1063',
                                     'P01': '219',
                                     'P02': '',
                                     'P03': '1329',
                                     'P04': '',
                                     'P05': '436',
                                     'worker_id': 'W00'}},
                         {'source_row': 6,
                          'values': {'P00': '642',
                                     'P01': '',
                                     'P02': '1130',
                                     'P03': '133',
                                     'P04': '199',
                                     'P05': '311',
                                     'worker_id': 'W04'}},
                         {'source_row': 7,
                          'values': {'P00': '102',
                                     'P01': '353',
                                     'P02': '651',
                                     'P03': '102',
                                     'P04': '',
                                     'P05': '7',
                                     'worker_id': 'W12'}}],
             'returned_rows': 8,
             'role': 'descriptive non-unique role',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
SKILL_MAP = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}

def solve_problem():
    tables = CSVQA_DATA['tables']
    worker_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    workers = []
    skill_lvl = {}
    for rec in worker_table['records']:
        w_id = rec['values']['worker_id']
        skill_str = rec['values']['skill']
        if skill_str not in SKILL_MAP:
            raise ValueError(f"Unknown skill level '{skill_str}' for worker {w_id}")
        workers.append(w_id)
        skill_lvl[w_id] = SKILL_MAP[skill_str]
    project_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    projects = []
    req_lvl = {}
    for rec in project_table['records']:
        p_id = rec['values']['project_id']
        skill_str = rec['values']['required_skill']
        if skill_str not in SKILL_MAP:
            raise ValueError(f"Unknown required skill '{skill_str}' for project {p_id}")
        projects.append(p_id)
        req_lvl[p_id] = SKILL_MAP[skill_str]
    cost_table = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    cost_rows = {rec['values']['worker_id']: rec['values'] for rec in cost_table['records']}
    for w in workers:
        if w not in cost_rows:
            raise ValueError(f'Cost row missing for worker {w}')
    cost_columns = [col for col in cost_table['columns'] if col != 'worker_id']
    for p in projects:
        if p not in cost_columns:
            raise ValueError(f'Cost column missing for project {p}')
    cost = {}
    allowed = {}
    for w in workers:
        row = cost_rows[w]
        cost[w] = {}
        allowed[w] = {}
        for p in projects:
            val = row.get(p, '')
            if val == '':
                cost[w][p] = 0
                allowed[w][p] = 0
            else:
                try:
                    c_val = int(val)
                except ValueError:
                    raise ValueError(f"Invalid cost value '{val}' for worker {w}, project {p}")
                cost[w][p] = c_val
                allowed[w][p] = 1 if skill_lvl[w] >= req_lvl[p] else 0
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(w, p) for w in workers for p in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((cost[w][p] * x[w, p] for w in workers for p in projects)), GRB.MINIMIZE)
    for p in projects:
        m.addConstr(gurobipy.quicksum((x[w, p] for w in workers)) == 1, name=f'proj_{p}')
    for w in workers:
        m.addConstr(gurobipy.quicksum((x[w, p] for p in projects)) <= 1, name=f'worker_{w}')
    for w in workers:
        for p in projects:
            if allowed[w][p] == 0:
                m.addConstr(x[w, p] == 0, name=f'forbid_{w}_{p}')
    m.optimize()
    return m
m = solve_problem()