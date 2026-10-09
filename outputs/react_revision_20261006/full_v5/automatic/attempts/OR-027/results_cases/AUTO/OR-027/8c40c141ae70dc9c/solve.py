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
                                         'value': 'Organ'}],
                         'logic': 'and'},
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

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        vals = rec['values']
        subcat = vals['Sub Category']
        items.append(subcat)
        try:
            revenue[subcat] = float(vals['Revenue'])
        except Exception:
            raise ValueError(f'Missing or invalid Revenue for {subcat}')
        try:
            demand[subcat] = int(vals['Demand'])
        except Exception:
            raise ValueError(f'Missing or invalid Demand for {subcat}')
        try:
            inventory[subcat] = float(vals['Initial Inventory'])
        except Exception:
            raise ValueError(f'Missing or invalid Initial Inventory for {subcat}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing data for item {i}')
    m = gp.Model('Supermart_Organ_Revenue')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()