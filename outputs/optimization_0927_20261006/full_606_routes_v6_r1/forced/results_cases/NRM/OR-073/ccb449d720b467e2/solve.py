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
table = [r['values'] for r in CSVQA_DATA['tables'][0]['records']]
products = ['Product I', 'Product II', 'Product III']
equip_A = ['A1', 'A2']
equip_B = ['B1', 'B2', 'B3']
EA_p = {'Product I': ['A1', 'A2'], 'Product II': ['A1', 'A2'], 'Product III': ['A2']}
EB_p = {'Product I': ['B1', 'B2', 'B3'], 'Product II': ['B1'], 'Product III': ['B2']}
equip_rows = {}
for (i, row) in enumerate(table):
    eq = row['Equipment / Cost']
    if eq in equip_A + equip_B:
        equip_rows[eq] = i
t_ep = {}
for e in equip_A + equip_B:
    t_ep[e] = {}
    row = table[equip_rows[e]]
    for p in products:
        val = row[p]
        if val != '':
            t_ep[e][p] = float(val)
T_e = {}
C_e = {}
for e in equip_A + equip_B:
    row = table[equip_rows[e]]
    T_e[e] = float(row['Available Equipment Operating Time'])
    C_e[e] = float(row['Equipment Cost at Full Load (yuan)'])
for row in table:
    if row['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)':
        c_p = {p: float(row[p]) for p in products}
    if row['Equipment / Cost'] == 'Unit Price (yuan/unit)':
        s_p = {p: float(row[p]) for p in products}
for p in products:
    for e in EA_p[p]:
        if e not in t_ep or p not in t_ep[e]:
            raise ValueError(f'Missing processing time for {e}, {p} in procedure A')
    for e in EB_p[p]:
        if e not in t_ep or p not in t_ep[e]:
            raise ValueError(f'Missing processing time for {e}, {p} in procedure B')
for e in equip_A + equip_B:
    if e not in T_e or e not in C_e:
        raise ValueError(f'Missing T_e or C_e for {e}')
m = gp.Model('factory_production')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(((e, p) for p in products for e in EA_p[p]), lb=0, vtype=GRB.CONTINUOUS, name='')
z_vars = m.addVars(((e, p) for p in products for e in EB_p[p]), lb=0, vtype=GRB.CONTINUOUS, name='')
profit = gp.quicksum((s_p[p] * x_vars[p] for p in products)) - gp.quicksum((c_p[p] * x_vars[p] for p in products)) - gp.quicksum((C_e[e] / T_e[e] * (gp.quicksum((t_ep[e][p] * y_vars[e, p] for p in products if (e, p) in y_vars)) + gp.quicksum((t_ep[e][p] * z_vars[e, p] for p in products if (e, p) in z_vars))) for e in equip_A + equip_B))
m.setObjective(profit, GRB.MAXIMIZE)
for p in products:
    m.addConstr(gp.quicksum((y_vars[e, p] for e in EA_p[p])) == x_vars[p], name=f'assignA_{p}')
for p in products:
    m.addConstr(gp.quicksum((z_vars[e, p] for e in EB_p[p])) == x_vars[p], name=f'assignB_{p}')
for e in equip_A + equip_B:
    m.addConstr(gp.quicksum((t_ep[e][p] * y_vars[e, p] for p in products if (e, p) in y_vars)) + gp.quicksum((t_ep[e][p] * z_vars[e, p] for p in products if (e, p) in z_vars)) <= T_e[e], name=f'time_{e}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')