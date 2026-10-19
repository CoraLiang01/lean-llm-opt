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
 'route': 'TP',
 'tables': [{'aliases': ['file_0_view_0', '43.csv'],
             'columns': ['Equipment / Cost',
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
import gurobipy as gp
from gurobipy import GRB
records = CSVQA_DATA['tables'][0]['records']
available_time = {}
cost_per_hour = {}
processing_time_A = {}
processing_time_B = {}
raw_material_cost = {}
unit_price = {}
A_equipment = []
B_equipment = []
for rec in records:
    equip = rec['values']['Equipment / Cost']
    if equip.startswith('A'):
        A_equipment.append(equip)
        available_time[equip] = float(rec['values']['Available Equipment Operating Time'])
        cost_per_hour[equip] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
        processing_time_A[equip] = {}
        for prod in ['Product I', 'Product II', 'Product III']:
            val = rec['values'][prod]
            if val != '':
                processing_time_A[equip][prod] = float(val)
    elif equip.startswith('B'):
        B_equipment.append(equip)
        available_time[equip] = float(rec['values']['Available Equipment Operating Time'])
        cost_per_hour[equip] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
        processing_time_B[equip] = {}
        for prod in ['Product I', 'Product II', 'Product III']:
            val = rec['values'][prod]
            if val != '':
                processing_time_B[equip][prod] = float(val)
    elif equip == 'Raw Material Cost (yuan/unit)':
        raw_material_cost['I'] = float(rec['values']['Product I'])
        raw_material_cost['II'] = float(rec['values']['Product II'])
        raw_material_cost['III'] = float(rec['values']['Product III'])
    elif equip == 'Unit Price (yuan/unit)':
        unit_price['I'] = float(rec['values']['Product I'])
        unit_price['II'] = float(rec['values']['Product II'])
        unit_price['III'] = float(rec['values']['Product III'])
products = ['I', 'II', 'III']
m = gp.Model('production')
x = {}
y = {}
for p in products:
    for a in A_equipment:
        if p in processing_time_A[a]:
            var = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
            x[p, a] = var
    for b in B_equipment:
        if p in processing_time_B[b]:
            var = m.addVar(vtype=GRB.CONTINUOUS, lb=0)
            y[p, b] = var
m.update()
for p in products:
    lhs = gp.quicksum((x[p, a] for a in A_equipment if (p, a) in x))
    rhs = gp.quicksum((y[p, b] for b in B_equipment if (p, b) in y))
    m.addConstr(lhs == rhs)
for a in A_equipment:
    expr = gp.quicksum((processing_time_A[a][p] * x[p, a] for p in products if (p, a) in x))
    m.addConstr(expr <= available_time[a])
for b in B_equipment:
    expr = gp.quicksum((processing_time_B[b][p] * y[p, b] for p in products if (p, b) in y))
    m.addConstr(expr <= available_time[b])
profit = gp.quicksum(((unit_price[p] - raw_material_cost[p]) * gp.quicksum((x[p, a] for a in A_equipment if (p, a) in x)) for p in products))
cost_A = gp.quicksum((cost_per_hour[a] * gp.quicksum((processing_time_A[a][p] * x[p, a] for p in products if (p, a) in x)) for a in A_equipment))
cost_B = gp.quicksum((cost_per_hour[b] * gp.quicksum((processing_time_B[b][p] * y[p, b] for p in products if (p, b) in y)) for b in B_equipment))
m.setObjective(profit - cost_A - cost_B, GRB.MAXIMIZE)
m.Params.MIPGap = 1e-06
m.optimize()
if m.status == GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Solver status: {m.status}')