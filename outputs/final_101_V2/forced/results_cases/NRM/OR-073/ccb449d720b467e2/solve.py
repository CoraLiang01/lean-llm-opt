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
 'tables': [{'columns': ['Equipment / Cost',
                         'Product I',
                         'Product II',
                         'Product III',
                         'Available Equipment Operating Time',
                         'Equipment Cost at Full Load (yuan)'],
             'file_index': 0,
             'file_name': '43.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'equipment and product parameters',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
import re
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'A factory produces three products, I, II, and III. Each product goes through two processing procedures, A and B. The factory has two types of equipment, A1 and A2, to complete procedure A, and three types of equipment, B1, B2, and B3, to complete procedure B. Product I can be processed on either type of A equipment or any type of B equipment. Product II can be processed on any type of A equipment, but when completing procedure B, it can only be processed on B1 equipment. Product III can only be processed on A2 and B2 equipment. Given the processing time, raw material cost, product selling price, available equipment operating time, and equipment cost at full load for each type of equipment, as shown in 43.csv, determine the optimal production plan to maximize profit. All production quantities should be continuous.', 'relationships': [], 'route': 'NRM', 'tables': [{'columns': ['Equipment / Cost', 'Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)'], 'file_index': 0, 'file_name': '43.csv', 'filters': {'conditions': [], 'logic': 'and'}, 'original_rows': 9, 'records': [{'source_row': 0, 'values': {'Available Equipment Operating Time': '6000', 'Equipment / Cost': 'A1', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '5', 'Product II': '10', 'Product III': ''}}, {'source_row': 1, 'values': {'Available Equipment Operating Time': '10000', 'Equipment / Cost': 'A2', 'Equipment Cost at Full Load (yuan)': '321', 'Product I': '7', 'Product II': '9', 'Product III': '12'}}, {'source_row': 2, 'values': {'Available Equipment Operating Time': '8000', 'Equipment / Cost': 'A3', 'Equipment Cost at Full Load (yuan)': '203', 'Product I': '6', 'Product II': '11', 'Product III': '2'}}, {'source_row': 3, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B1', 'Equipment Cost at Full Load (yuan)': '250', 'Product I': '6', 'Product II': '8', 'Product III': ''}}, {'source_row': 4, 'values': {'Available Equipment Operating Time': '7000', 'Equipment / Cost': 'B2', 'Equipment Cost at Full Load (yuan)': '783', 'Product I': '4', 'Product II': '', 'Product III': '11'}}, {'source_row': 5, 'values': {'Available Equipment Operating Time': '4000', 'Equipment / Cost': 'B3', 'Equipment Cost at Full Load (yuan)': '200', 'Product I': '7', 'Product II': '', 'Product III': ''}}, {'source_row': 6, 'values': {'Available Equipment Operating Time': '5000', 'Equipment / Cost': 'B4', 'Equipment Cost at Full Load (yuan)': '300', 'Product I': '3', 'Product II': '5', 'Product III': '8'}}, {'source_row': 7, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Raw Material Cost (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '0.25', 'Product II': '0.35', 'Product III': '0.5'}}, {'source_row': 8, 'values': {'Available Equipment Operating Time': '', 'Equipment / Cost': 'Unit Price (yuan/unit)', 'Equipment Cost at Full Load (yuan)': '', 'Product I': '1.25', 'Product II': '2', 'Product III': '2.8'}}], 'returned_rows': 9, 'role': 'equipment and product parameters', 'table_id': 'file_0_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError('Required table file_0_view_0 not found.')
records = table['records']
columns = table['columns']
product_names = ['Product I', 'Product II', 'Product III']
equipment_rows = []
raw_material_row = None
unit_price_row = None
for rec in records:
    eq = rec['values']['Equipment / Cost']
    if eq == 'Raw Material Cost (yuan/unit)':
        raw_material_row = rec
    elif eq == 'Unit Price (yuan/unit)':
        unit_price_row = rec
    else:
        equipment_rows.append(rec)
if raw_material_row is None or unit_price_row is None:
    raise ValueError('Raw material cost or unit price row missing.')
P = product_names
A = []
B = []
E = []
for rec in equipment_rows:
    eq = rec['values']['Equipment / Cost']
    if re.match('^A\\d+$', eq):
        A.append(eq)
        E.append(eq)
    elif re.match('^B\\d+$', eq):
        B.append(eq)
        E.append(eq)
    else:
        continue
A = [e for e in A if e in ['A1', 'A2']]
B = [e for e in B if e in ['B1', 'B2', 'B3']]
E = A + B
t_ap = {}
t_bp = {}
T_e = {}
C_e = {}
delta_ap = {}
delta_bp = {}
for rec in equipment_rows:
    eq = rec['values']['Equipment / Cost']
    if eq in A:
        for p in P:
            val = rec['values'][p]
            if val.strip() != '':
                t_ap[eq, p] = float(val)
                delta_ap[eq, p] = 1
            else:
                t_ap[eq, p] = 0.0
                delta_ap[eq, p] = 0
        T_e[eq] = float(rec['values']['Available Equipment Operating Time'])
        C_e[eq] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
    elif eq in B:
        for p in P:
            val = rec['values'][p]
            if val.strip() != '':
                t_bp[eq, p] = float(val)
                delta_bp[eq, p] = 1
            else:
                t_bp[eq, p] = 0.0
                delta_bp[eq, p] = 0
        T_e[eq] = float(rec['values']['Available Equipment Operating Time'])
        C_e[eq] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
c_p = {}
s_p = {}
for p in P:
    c_p[p] = float(raw_material_row['values'][p])
    s_p[p] = float(unit_price_row['values'][p])
for a in A:
    for p in P:
        if (a, p) not in t_ap or (a, p) not in delta_ap:
            raise ValueError(f'Missing t_ap or delta_ap for {a}, {p}')
for b in B:
    for p in P:
        if (b, p) not in t_bp or (b, p) not in delta_bp:
            raise ValueError(f'Missing t_bp or delta_bp for {b}, {p}')
for e in E:
    if e not in T_e or e not in C_e:
        raise ValueError(f'Missing T_e or C_e for {e}')
for p in P:
    if p not in c_p or p not in s_p:
        raise ValueError(f'Missing c_p or s_p for {p}')
m = gp.Model('Original_RAG_NRM')
x = m.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(A, P, lb=0, vtype=GRB.CONTINUOUS, name='')
z = m.addVars(B, P, lb=0, vtype=GRB.CONTINUOUS, name='')
for p in P:
    m.addConstr(x[p] == gp.quicksum((y[a, p] for a in A)), name=f'assignA_{p}')
for a in A:
    for p in P:
        if delta_ap[a, p] == 0:
            m.addConstr(y[a, p] == 0, name=f'yzero_{a}_{p}')
for p in P:
    m.addConstr(x[p] == gp.quicksum((z[b, p] for b in B)), name=f'assignB_{p}')
for b in B:
    for p in P:
        if delta_bp[b, p] == 0:
            m.addConstr(z[b, p] == 0, name=f'zzero_{b}_{p}')
for a in A:
    m.addConstr(gp.quicksum((t_ap[a, p] * y[a, p] for p in P)) <= T_e[a], name=f'timecapA_{a}')
for b in B:
    m.addConstr(gp.quicksum((t_bp[b, p] * z[b, p] for p in P)) <= T_e[b], name=f'timecapB_{b}')
expr = gp.quicksum(((s_p[p] - c_p[p]) * x[p] for p in P))
for a in A:
    expr -= C_e[a] / T_e[a] * gp.quicksum((t_ap[a, p] * y[a, p] for p in P))
for b in B:
    expr -= C_e[b] / T_e[b] * gp.quicksum((t_bp[b, p] * z[b, p] for p in P))
m.setObjective(expr, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')