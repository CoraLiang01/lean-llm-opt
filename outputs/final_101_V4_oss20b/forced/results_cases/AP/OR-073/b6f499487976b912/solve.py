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
 'route': 'AP',
 'tables': [{'columns': ['Equipment / Cost',
                         'Product I',
                         'Product II',
                         'Product III',
                         'Available Equipment Operating Time',
                         'Equipment Cost at Full Load (yuan)'],
             'file_index': 0,
             'file_name': '43.csv',
             'filters': {},
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
                         {'source_row': 2,
                          'values': {'Available Equipment Operating Time': '8000',
                                     'Equipment / Cost': 'A3',
                                     'Equipment Cost at Full Load (yuan)': '203',
                                     'Product I': '6',
                                     'Product II': '11',
                                     'Product III': '2'}},
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
                                     'Product III': ''}},
                         {'source_row': 6,
                          'values': {'Available Equipment Operating Time': '5000',
                                     'Equipment / Cost': 'B4',
                                     'Equipment Cost at Full Load (yuan)': '300',
                                     'Product I': '3',
                                     'Product II': '5',
                                     'Product III': '8'}},
                         {'source_row': 7,
                          'values': {'Available Equipment Operating Time': '',
                                     'Equipment / Cost': 'Raw Material Cost (yuan/unit)',
                                     'Equipment Cost at Full Load (yuan)': '',
                                     'Product I': '0.25',
                                     'Product II': '0.35',
                                     'Product III': '0.5'}},
                         {'source_row': 8,
                          'values': {'Available Equipment Operating Time': '',
                                     'Equipment / Cost': 'Unit Price (yuan/unit)',
                                     'Equipment Cost at Full Load (yuan)': '',
                                     'Product I': '1.25',
                                     'Product II': '2',
                                     'Product III': '2.8'}}],
             'returned_rows': 9,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
data = CSVQA_DATA
table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
records = table['records']
products = ['I', 'II', 'III']
product_col = {p: f'Product {p}' for p in products}
equipment = []
a_e = {}
d_e = {}
t_e_p = {}
delta_e_p = {}
for rec in records:
    e_id = rec['values']['Equipment / Cost']
    if e_id in ['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)']:
        continue
    equipment.append(e_id)
    a_val = rec['values']['Available Equipment Operating Time']
    a_e[e_id] = float(a_val) if a_val else 0.0
    d_val = rec['values']['Equipment Cost at Full Load (yuan)']
    d_e[e_id] = float(d_val) if d_val else 0.0
    t_e_p[e_id] = {}
    delta_e_p[e_id] = {}
    for p in products:
        col = product_col[p]
        val = rec['values'][col]
        if val:
            t_e_p[e_id][p] = float(val)
            delta_e_p[e_id][p] = True
        else:
            t_e_p[e_id][p] = 0.0
            delta_e_p[e_id][p] = False
r_p = {}
for rec in records:
    if rec['values']['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)':
        for p in products:
            col = product_col[p]
            r_p[p] = float(rec['values'][col])
        break
c_p = {}
for rec in records:
    if rec['values']['Equipment / Cost'] == 'Unit Price (yuan/unit)':
        for p in products:
            col = product_col[p]
            c_p[p] = float(rec['values'][col])
        break
m = gurobipy.Model()
m.setParam('MIPGap', 1e-06)
x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars([(e, p) for e in equipment for p in products], lb=0, vtype=GRB.CONTINUOUS, name='')
for p in products:
    m.addConstr(x[p] == gurobipy.quicksum((y[e, p] for e in equipment)), name=f'prod_bal_{p}')
for e in equipment:
    m.addConstr(gurobipy.quicksum((t_e_p[e][p] * y[e, p] for p in products)) <= a_e[e], name=f'cap_{e}')
for e in equipment:
    for p in products:
        if not delta_e_p[e][p]:
            m.addConstr(y[e, p] == 0, name=f'elig_{e}_{p}')
obj = gurobipy.quicksum(((c_p[p] - r_p[p]) * x[p] for p in products)) - gurobipy.quicksum((d_e[e] * t_e_p[e][p] * y[e, p] for e in equipment for p in products))
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')