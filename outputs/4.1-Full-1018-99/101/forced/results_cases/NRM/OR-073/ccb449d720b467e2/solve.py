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
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Filter lacks query evidence: {'column': 'Equipment / Cost', 'operator': 'in', "
                                   "'dtype': 'string', 'value': ['A1', 'A2', 'B1', 'B2', 'B3'], 'evidence': 'A1 and A2 "
                                   "and B1 and B2 and B3'}",
                'planner_errors': ["Filter lacks query evidence: {'column': 'Equipment / Cost', 'operator': 'in', "
                                   "'dtype': 'string', 'value': ['A1', 'A2', 'B1', 'B2', 'B3'], 'evidence': 'A1 and A2 "
                                   "and B1 and B2 and B3'}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError('Required table file_0_view_0 not found.')
records = table['records']
product_names = ['Product I', 'Product II', 'Product III']
equipment_rows = []
raw_material_row = None
unit_price_row = None
for rec in records:
    row_name = rec['values']['Equipment / Cost']
    if row_name in ['A1', 'A2', 'B1', 'B2', 'B3']:
        equipment_rows.append(rec)
    elif row_name == 'Raw Material Cost (yuan/unit)':
        raw_material_row = rec
    elif row_name == 'Unit Price (yuan/unit)':
        unit_price_row = rec
if len(equipment_rows) != 5:
    raise ValueError('Did not find all required equipment rows (A1, A2, B1, B2, B3).')
if raw_material_row is None or unit_price_row is None:
    raise ValueError('Missing raw material or unit price row.')
P = product_names
E = [rec['values']['Equipment / Cost'] for rec in equipment_rows]
A = ['A1', 'A2']
B = ['B1', 'B2', 'B3']
t_ep = {}
for rec in equipment_rows:
    e = rec['values']['Equipment / Cost']
    for p in P:
        val = rec['values'][p]
        if val.strip() != '':
            t_ep[e, p] = float(val)
        else:
            t_ep[e, p] = None
T_e = {}
for rec in equipment_rows:
    e = rec['values']['Equipment / Cost']
    T_e[e] = float(rec['values']['Available Equipment Operating Time'])
C_e = {}
for rec in equipment_rows:
    e = rec['values']['Equipment / Cost']
    C_e[e] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
c_p = {}
for p in P:
    c_p[p] = float(raw_material_row['values'][p])
s_p = {}
for p in P:
    s_p[p] = float(unit_price_row['values'][p])
eligible_y = {}
for e in E:
    for p in P:
        if t_ep[e, p] is not None:
            eligible_y[e, p] = True
        else:
            eligible_y[e, p] = False
m = gp.Model('factory_production_plan')
x_vars = m.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars([(e, p) for (e, p), eligible in eligible_y.items() if eligible], lb=0, vtype=GRB.CONTINUOUS, name='')
m.addConstr((y_vars['A1', 'Product I'] if eligible_y['A1', 'Product I'] else 0) + (y_vars['A2', 'Product I'] if eligible_y['A2', 'Product I'] else 0) == x_vars['Product I'])
m.addConstr((y_vars['A1', 'Product II'] if eligible_y['A1', 'Product II'] else 0) + (y_vars['A2', 'Product II'] if eligible_y['A2', 'Product II'] else 0) == x_vars['Product II'])
m.addConstr((y_vars['A2', 'Product III'] if eligible_y['A2', 'Product III'] else 0) == x_vars['Product III'])
m.addConstr((y_vars['B1', 'Product I'] if eligible_y['B1', 'Product I'] else 0) + (y_vars['B2', 'Product I'] if eligible_y['B2', 'Product I'] else 0) + (y_vars['B3', 'Product I'] if eligible_y['B3', 'Product I'] else 0) == x_vars['Product I'])
m.addConstr((y_vars['B1', 'Product II'] if eligible_y['B1', 'Product II'] else 0) == x_vars['Product II'])
m.addConstr((y_vars['B2', 'Product III'] if eligible_y['B2', 'Product III'] else 0) == x_vars['Product III'])
for e in E:
    expr = gp.LinExpr()
    for p in P:
        if eligible_y[e, p]:
            expr += t_ep[e, p] * y_vars[e, p]
    m.addConstr(expr <= T_e[e])
profit_expr = gp.quicksum((s_p[p] * x_vars[p] for p in P)) - gp.quicksum((c_p[p] * x_vars[p] for p in P)) - gp.quicksum((C_e[e] / T_e[e] * gp.quicksum((t_ep[e, p] * y_vars[e, p] for p in P if eligible_y[e, p])) for e in E))
m.setObjective(profit_expr, GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')