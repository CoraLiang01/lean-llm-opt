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
 'tables': [{'aliases': ['file_0_view_0', '43.csv'],
             'columns': ['Equipment / Cost',
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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
records = CSVQA_DATA['tables'][0]['records']
products = ['I', 'II', 'III']
equipments = []
available_time = {}
equipment_cost = {}
proc_time = {}
raw_material_cost = {}
unit_price = {}
for rec in records:
    vals = rec['values']
    equip = vals['Equipment / Cost']
    if equip == 'Raw Material Cost (yuan/unit)':
        for p in products:
            raw_material_cost[p] = float(vals[f'Product {p}'])
    elif equip == 'Unit Price (yuan/unit)':
        for p in products:
            unit_price[p] = float(vals[f'Product {p}'])
    else:
        equipments.append(equip)
        available_time[equip] = float(vals['Available Equipment Operating Time'])
        equipment_cost[equip] = float(vals['Equipment Cost at Full Load (yuan)'])
        proc_time[equip] = {}
        for p in products:
            val = vals[f'Product {p}']
            if val != '':
                proc_time[equip][p] = float(val)
allowedA = {'I': ['A1', 'A2', 'A3'], 'II': ['A1', 'A2', 'A3'], 'III': ['A2']}
allowedB = {'I': ['B1', 'B2', 'B3', 'B4'], 'II': ['B1'], 'III': ['B2']}
m = gp.Model('Production_Optimization')
Q = {}
for p in products:
    Q[p] = m.addVar(vtype=GRB.CONTINUOUS)
xA = {}
xB = {}
for p in products:
    for e in allowedA[p]:
        xA[p, e] = m.addVar(vtype=GRB.CONTINUOUS)
    for e in allowedB[p]:
        xB[p, e] = m.addVar(vtype=GRB.CONTINUOUS)
for p in products:
    m.addConstr(Q[p] == gp.quicksum((xA[p, e] for e in allowedA[p])))
    m.addConstr(Q[p] == gp.quicksum((xB[p, e] for e in allowedB[p])))
for e in equipments:
    if e in allowedA['I'] or e in allowedA['II'] or e in allowedA['III']:
        m.addConstr(gp.quicksum((proc_time[e][p] * xA[p, e] for p in products if (p, e) in xA)), '<=', available_time[e])
    if e in allowedB['I'] or e in allowedB['II'] or e in allowedB['III']:
        m.addConstr(gp.quicksum((proc_time[e][p] * xB[p, e] for p in products if (p, e) in xB)), '<=', available_time[e])
profit_per_unit = {p: unit_price[p] - raw_material_cost[p] for p in products}
obj = gp.quicksum((profit_per_unit[p] * Q[p] for p in products))
cost = gp.quicksum((equipment_cost[e] * (gp.quicksum((proc_time[e][p] * xA[p, e] for p in products if (p, e) in xA)) + gp.quicksum((proc_time[e][p] * xB[p, e] for p in products if (p, e) in xB))) for e in equipments))
m.setObjective(obj - cost, GRB.MAXIMIZE)
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective Value:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver Status:', m.Status)