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
                                         'evidence': 'products classified under ‘Organ’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Organic'}],
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
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = CSVQA_DATA['tables'][0]
records = table['records']
I = []
A = {}
d = {}
I_param = {}
for rec in records:
    vals = rec['values']
    subcat = vals['Sub Category']
    if not subcat.startswith('Organic'):
        continue
    idx = subcat
    I.append(idx)
    try:
        A[idx] = float(vals['Revenue'])
        d[idx] = int(vals['Demand'])
        I_param[idx] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for {idx}: {e}')
if not set(A.keys()) == set(d.keys()) == set(I_param.keys()) == set(I):
    raise ValueError('Mismatch in index sets for parameters.')
m = gp.Model('Supermart_Organ_RevOpt')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I_param[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')