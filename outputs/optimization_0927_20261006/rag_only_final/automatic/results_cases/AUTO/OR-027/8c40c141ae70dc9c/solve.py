CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'filters': {'conditions': [{'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': "'Organ' product",
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Organ'},
                                        {'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': "'Organ' product",
                                         'inclusive': 'both',
                                         'operator': 'contains',
                                         'value': 'Organ'}],
                         'logic': 'or'},
             'original_rows': 23,
             'records': [{'source_row': 17,
                          'values': {'Demand': '678906',
                                     'Initial Inventory': '5034020.0',
                                     'Revenue': '60.8',
                                     'Sub Category': 'Organic Fruits'}},
                         {'source_row': 18,
                          'values': {'Demand': '749927',
                                     'Initial Inventory': '5589290.0',
                                     'Revenue': '918.45',
                                     'Sub Category': 'Organic Staples'}},
                         {'source_row': 19,
                          'values': {'Demand': '699808',
                                     'Initial Inventory': '5202710.0',
                                     'Revenue': '77.52',
                                     'Sub Category': 'Organic Vegetables'}}],
             'returned_rows': 3,
             'role': 'product revenue, demand, and inventory parameters',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB, quicksum
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table['records']
I = []
Revenue = {}
Initial_Inventory = {}
Demand = {}
for rec in records:
    vals = rec['values']
    subcat = vals['Sub Category']
    if 'Organ' in subcat:
        idx = subcat
        I.append(idx)
        try:
            Revenue[idx] = float(vals['Revenue'])
            Initial_Inventory[idx] = float(vals['Initial Inventory'])
            Demand[idx] = float(vals['Demand'])
        except Exception as e:
            raise RuntimeError(f'Error parsing numeric fields for {idx}: {e}')
for idx in I:
    if idx not in Revenue or idx not in Initial_Inventory or idx not in Demand:
        raise RuntimeError(f'Missing parameter for product {idx}')
m = Model()
m.setParam('MIPGap', 0.0001)
x_vars = m.addVars(I, lb=0, ub={i: min(Initial_Inventory[i], Demand[i]) for i in I}, vtype=GRB.CONTINUOUS, name='')
inv_constrs = m.addConstrs((x_vars[i] <= Initial_Inventory[i] for i in I), name='')
dem_constrs = m.addConstrs((x_vars[i] <= Demand[i] for i in I), name='')
m.setObjective(quicksum((Revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in I:
        print(f'{x_vars[i].VarName}: {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')