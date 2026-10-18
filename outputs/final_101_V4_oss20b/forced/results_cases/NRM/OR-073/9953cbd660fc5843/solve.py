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
 'route': 'NRM',
 'tables': [{'aliases': ['file_0_view_0', '43.csv'],
             'columns': ['Equipment / Cost',
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
                                                     'B.',
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
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
products = ['I', 'II', 'III']
procedures = ['A', 'B']
equipment = ['A1', 'A2', 'B1', 'B2', 'B3']
prod_col_map = {'Product I': 'I', 'Product II': 'II', 'Product III': 'III'}
t = {p: {r: {} for r in procedures} for p in products}
T = {}
cE = {}
cRM = {p: 0.0 for p in products}
p_price = {p: 0.0 for p in products}
for rec in records:
    e = rec['values']['Equipment / Cost']
    T[e] = float(rec['values']['Available Equipment Operating Time'])
    cE[e] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
    proc = 'A' if e in ['A1', 'A2'] else 'B'
    for col, p in prod_col_map.items():
        val = rec['values'][col]
        if val != '':
            t[p][proc][e] = float(val)
m = gp.Model()
m.setParam(GRB.Param.MIPGap, 1e-06)
y = {p: m.addVar(vtype=GRB.CONTINUOUS) for p in products}
x = {}
for p in products:
    for r in procedures:
        for e in equipment:
            if e in t[p][r]:
                x[p, r, e] = m.addVar(vtype=GRB.CONTINUOUS)
for p in products:
    m.addConstr(gp.quicksum((x[p, 'A', e] for e in equipment if (p, 'A', e) in x)) == y[p])
    m.addConstr(gp.quicksum((x[p, 'B', e] for e in equipment if (p, 'B', e) in x)) == y[p])
for e in equipment:
    expr = gp.quicksum((t[p][r][e] * x[p, r, e] for p in products for r in procedures if (p, r, e) in x))
    m.addConstr(expr <= T[e])
obj = gp.quicksum(((p_price[p] - cRM[p]) * y[p] for p in products))
obj -= gp.quicksum((cE[e] * t[p][r][e] * x[p, r, e] for p in products for r in procedures for e in equipment if (p, r, e) in x))
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal:', m.ObjVal)
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print('Status:', m.Status)