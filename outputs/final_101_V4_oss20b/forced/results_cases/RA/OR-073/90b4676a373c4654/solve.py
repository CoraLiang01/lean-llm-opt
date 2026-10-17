CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A '
          'and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of '
          'equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A '
          'equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when '
          'completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 '
          'and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment '
          'operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine '
          'the optimal production plan to maximize profit. All production quantities should be continuous.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Equipment / Cost',
                         'Product I',
                         'Product II',
                         'Product III',
                         'Available Equipment Operating Time',
                         'Equipment Cost at Full Load (yuan)'],
             'file_index': 0,
             'file_name': '43.csv',
             'filters': {'conditions': [{'column': 'Equipment / Cost',
                                         'dtype': 'string',
                                         'evidence': 'two types of equipment, A1 and A2, to complete procedure A, and '
                                                     'three types of equipment, B1, B2, and B3, to complete procedure '
                                                     'B',
                                         'operator': 'in',
                                         'value': ['A1', 'A2', 'B1', 'B2', 'B3']}],
                         'logic': 'and'},
             'original_rows': 9,
             'records': [{'source_row': 0,
                          'values': {'Available Equipment Operating Time': '6000',
                                     'Equipment / Cost': 'A1',
                                     'Equipment Cost at Full Load (yuan)': '300',
                                     'Product I': '5',
                                     'Product II': '10',
                                     'Product III': ''}},
                         {'source_row': 1,
                          'values': {'Available Equipment Operating Time': '10000',
                                     'Equipment / Cost': 'A2',
                                     'Equipment Cost at Full Load (yuan)': '321',
                                     'Product I': '7',
                                     'Product II': '9',
                                     'Product III': '12'}},
                         {'source_row': 3,
                          'values': {'Available Equipment Operating Time': '4000',
                                     'Equipment / Cost': 'B1',
                                     'Equipment Cost at Full Load (yuan)': '250',
                                     'Product I': '6',
                                     'Product II': '8',
                                     'Product III': ''}},
                         {'source_row': 4,
                          'values': {'Available Equipment Operating Time': '7000',
                                     'Equipment / Cost': 'B2',
                                     'Equipment Cost at Full Load (yuan)': '783',
                                     'Product I': '4',
                                     'Product II': '',
                                     'Product III': '11'}},
                         {'source_row': 5,
                          'values': {'Available Equipment Operating Time': '4000',
                                     'Equipment / Cost': 'B3',
                                     'Equipment Cost at Full Load (yuan)': '200',
                                     'Product I': '7',
                                     'Product II': '',
                                     'Product III': ''}}],
             'returned_rows': 5,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
table = None
for tbl in CSVQA_DATA['tables']:
    if tbl.get('table_id') == 'file_0_view_0':
        table = tbl
        break
if table is None:
    raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
records = table['records']
product_cols = ['Product I', 'Product II', 'Product III']
product_keys = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
products = ['I', 'II', 'III']
avail_e = {}
cost_e = {}
time_p_e = {}
for rec in records:
    vals = rec['values']
    e = vals['Equipment / Cost'].strip()
    if not e:
        continue
    if 'Available Equipment Operating Time' not in vals:
        raise ValueError(f"Missing 'Available Equipment Operating Time' for equipment {e}")
    avail = float(vals['Available Equipment Operating Time'].strip())
    avail_e[e] = avail
    if 'Equipment Cost at Full Load (yuan)' not in vals:
        raise ValueError(f"Missing 'Equipment Cost at Full Load (yuan)' for equipment {e}")
    cost = float(vals['Equipment Cost at Full Load (yuan)'].strip())
    cost_e[e] = cost
    for col in product_cols:
        val = vals.get(col, '').strip()
        if val:
            p = product_keys[col]
            time = float(val)
            time_p_e[p, e] = time
admissible_pairs = [(p, e) for p, e in time_p_e.keys()]
admissible_p_by_e = {}
for e in avail_e:
    admissible_p_by_e[e] = [p for p, eq in admissible_pairs if eq == e]
profit_p = {p: 0.0 for p in products}
m = gurobipy.Model()
m.setParam('MIPGap', 1e-06)
x = m.addVars(admissible_pairs, lb=0, vtype=GRB.CONTINUOUS, name='')
obj_expr = gurobipy.quicksum(((profit_p[p] - cost_e[e] * time_p_e[p, e]) * x[p, e] for p, e in admissible_pairs))
m.setObjective(obj_expr, GRB.MAXIMIZE)
m.addConstrs((gurobipy.quicksum((time_p_e[p, e] * x[p, e] for p in admissible_p_by_e[e])) <= avail_e[e] for e in avail_e), name='')
m.optimize()