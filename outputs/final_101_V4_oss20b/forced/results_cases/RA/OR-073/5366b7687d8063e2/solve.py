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
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
table_records = CSVQA_DATA['tables'][0]['records']
df = pd.DataFrame([rec['values'] for rec in table_records])
numeric_cols = ['Product I', 'Product II', 'Product III', 'Available Equipment Operating Time', 'Equipment Cost at Full Load (yuan)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
equip_rows = df[~df['Equipment / Cost'].isin(['Raw Material Cost (yuan/unit)', 'Unit Price (yuan/unit)'])]
price_row = df[df['Equipment / Cost'] == 'Unit Price (yuan/unit)'].iloc[0]
cost_row = df[df['Equipment / Cost'] == 'Raw Material Cost (yuan/unit)'].iloc[0]
equipment_data = {}
for _, row in equip_rows.iterrows():
    equip = row['Equipment / Cost']
    equipment_data[equip] = {'times': {'I': row['Product I'], 'II': row['Product II'], 'III': row['Product III']}, 'available_time': row['Available Equipment Operating Time'], 'cost': row['Equipment Cost at Full Load (yuan)']}
revenue_per_product = {'I': price_row['Product I'] - cost_row['Product I'], 'II': price_row['Product II'] - cost_row['Product II'], 'III': price_row['Product III'] - cost_row['Product III']}
allowed_combos = [('I', 'A1'), ('I', 'A2'), ('I', 'A3'), ('I', 'B1'), ('I', 'B2'), ('I', 'B3'), ('I', 'B4'), ('II', 'A1'), ('II', 'A2'), ('II', 'A3'), ('II', 'B1'), ('II', 'B4'), ('III', 'A2'), ('III', 'B2')]
m = gp.Model('production')
m.Params.MIPGap = 1e-06
var_dict = {}
for prod, equip in allowed_combos:
    var_name = f'x_{prod}_{equip}'
    var_dict[prod, equip] = m.addVar(vtype=GRB.CONTINUOUS)
m.update()
obj_terms = []
for (prod, equip), var in var_dict.items():
    if prod == 'II' and equip == 'B4':
        continue
    time = equipment_data[equip]['times'][prod]
    cost_per_unit = equipment_data[equip]['cost'] * time
    coef = revenue_per_product[prod] - cost_per_unit
    obj_terms.append(coef * var)
m.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)
for equip, data in equipment_data.items():
    terms = []
    for prod in ['I', 'II', 'III']:
        time = data['times'][prod]
        if pd.notna(time):
            var = var_dict.get((prod, equip))
            if var is not None:
                terms.append(time * var)
    if terms:
        m.addConstr(gp.quicksum(terms) <= data['available_time'])
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(f'{var.VarName} {var.X}')
else:
    print(m.Status)