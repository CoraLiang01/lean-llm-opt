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
 'route': 'FLP',
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
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
table = CSVQA_DATA['tables'][0]
records = table['records']
products = ['I', 'II', 'III']
equipment = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4']
processing_time = {}
available_time = {}
equipment_cost = {}
raw_material_cost = {}
unit_price = {}
for rec in records:
    equip = rec['values']['Equipment / Cost']
    if equip in equipment:
        available_time[equip] = float(rec['values']['Available Equipment Operating Time'])
        equipment_cost[equip] = float(rec['values']['Equipment Cost at Full Load (yuan)'])
        for p in products:
            val = rec['values'][f'Product {p}']
            if val != '':
                processing_time.setdefault(equip, {})[p] = float(val)
    elif equip == 'Raw Material Cost (yuan/unit)':
        for p in products:
            raw_material_cost[p] = float(rec['values'][f'Product {p}'])
    elif equip == 'Unit Price (yuan/unit)':
        for p in products:
            unit_price[p] = float(rec['values'][f'Product {p}'])
m = gp.Model('ProductionPlan')
x = m.addVars(products, equipment, lb=0)
y = m.addVars(equipment, vtype=GRB.BINARY)
profit_per_unit = {p: unit_price[p] - raw_material_cost[p] for p in products}
m.setObjective(gp.quicksum((profit_per_unit[p] * x[p, e] for p in products for e in equipment)) - gp.quicksum((equipment_cost[e] * y[e] for e in equipment)), GRB.MAXIMIZE)
for e in equipment:
    m.addConstr(gp.quicksum((processing_time[e][p] * x[p, e] for p in products if p in processing_time[e])) <= available_time[e] * y[e])
eligibility_zero = [('I', 'A3'), ('III', 'A1'), ('III', 'A3'), ('III', 'B1'), ('III', 'B3'), ('III', 'B4'), ('II', 'B2'), ('II', 'B3'), ('II', 'B4')]
for p, e in eligibility_zero:
    m.addConstr(x[p, e] == 0)
for e in equipment:
    for p in products:
        if p not in processing_time.get(e, {}):
            m.addConstr(x[p, e] == 0)
m.Params.MIPGap = 1e-06
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('Objective:', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Status:', m.Status)