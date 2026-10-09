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
                                         'value': 'Organ'},
                                        {'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Organ’',
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
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB

def solve_organ_revenue_optimization(CSVQA_DATA):
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
    I = []
    Revenue = {}
    InitialInventory = {}
    Demand = {}
    for rec in table['records']:
        vals = rec['values']
        subcat = vals['Sub Category']
        if re.match('^Organ', subcat) or re.search('Organ', subcat):
            i = subcat
            I.append(i)
            try:
                Revenue_i = float(vals['Revenue'])
                InitialInventory_i = float(vals['Initial Inventory'])
                Demand_i = float(vals['Demand'])
            except Exception as e:
                raise RuntimeError(f"Malformed or missing data for product '{i}': {e}")
            Revenue[i] = Revenue_i
            InitialInventory[i] = InitialInventory_i
            Demand[i] = Demand_i
    for i in I:
        if i not in Revenue or i not in InitialInventory or i not in Demand:
            raise RuntimeError(f"Missing parameter(s) for product '{i}'.")
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(sum((Revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= InitialInventory[i], name='')
        m.addConstr(x[i] <= Demand[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName}: {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_organ_revenue_optimization(CSVQA_DATA)