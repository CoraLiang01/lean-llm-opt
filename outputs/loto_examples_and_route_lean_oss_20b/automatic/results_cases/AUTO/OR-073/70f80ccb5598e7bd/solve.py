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
table = CSVQA_DATA['tables'][0]
records = table['records']
products = ['I', 'II', 'III']
equipment = []
available_time = {}
equipment_cost_per_hour = {}
time_per_unit = {p: {} for p in products}
raw_material_cost = {}
unit_price = {}
for rec in records:
    vals = rec['values']
    equip_name = vals['Equipment / Cost']
    if equip_name in ('Raw Material Cost (yuan)', 'Unit Price (yuan/unit)'):
        if equip_name == 'Raw Material Cost (yuan)':
            for p in products:
                raw_material_cost[p] = float(vals[f'Product {p}'])
        else:
            for p in products:
                unit_price[p] = float(vals[f'Product {p}'])
        continue
    equipment.append(equip_name)
    available_time[equip_name] = float(vals['Available Equipment Operating Time'])
    equipment_cost_per_hour[equip_name] = float(vals['Equipment Cost at Full Load (yuan)'])
    for p in products:
        val = vals[f'Product {p}']
        if val != '':
            time_per_unit[p][equip_name] = float(val)
        else:
            time_per_unit[p][equip_name] = None
m = gp.Model('production')
x = {}
for p in products:
    for e in equipment:
        if time_per_unit[p][e] is not None:
            var = m.addVar(vtype=GRB.CONTINUOUS)
            x[p, e] = var
m.update()
obj_terms = []
for (p, e), var in x.items():
    profit_per_unit = unit_price[p] - raw_material_cost[p] - equipment_cost_per_hour[e] * time_per_unit[p][e]
    obj_terms.append(profit_per_unit * var)
m.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)
for e in equipment:
    expr = gp.quicksum((time_per_unit[p][e] * x[p, e] for p in products if time_per_unit[p][e] is not None))
    m.addConstr(expr <= available_time[e])
m.setParam('MIPGap', 1e-06)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'Objective value: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName} = {var.X}')
else:
    print(f'Status: {m.Status}')