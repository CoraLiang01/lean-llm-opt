CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail establishment seeks to optimize the allocation of its merchandise across distinct categories '
          '(electronics, apparel, homeware, etc.) to achieve the highest possible total revenue. Each category '
          'maintains its own demand level, with revenue figures provided in the ‘Revenue’ column and current stock '
          'quantities documented in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The optimization challenge involves '
          'determining the ideal fulfillment quantities x_i for each product that allocate available inventory while '
          'respecting inventory limits and maximizing total revenue.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'electronics',
                                         'operator': 'contains',
                                         'value': 'electronics'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'apparel',
                                         'operator': 'contains',
                                         'value': 'apparel'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'homeware',
                                         'operator': 'contains',
                                         'value': 'homeware'}],
                         'logic': 'or'},
             'original_rows': 35,
             'records': [{'source_row': 10,
                          'values': {'Demand': '273',
                                     'Initial Inventory': '1810',
                                     'Product Name': 'Electronics - 25',
                                     'Revenue': '25'}},
                         {'source_row': 11,
                          'values': {'Demand': '220',
                                     'Initial Inventory': '1410',
                                     'Product Name': 'Electronics - 30',
                                     'Revenue': '30'}},
                         {'source_row': 12,
                          'values': {'Demand': '286',
                                     'Initial Inventory': '1830',
                                     'Product Name': 'Electronics - 300',
                                     'Revenue': '300'}},
                         {'source_row': 13,
                          'values': {'Demand': '268',
                                     'Initial Inventory': '1750',
                                     'Product Name': 'Electronics - 50',
                                     'Revenue': '50'}},
                         {'source_row': 14,
                          'values': {'Demand': '262',
                                     'Initial Inventory': '1690',
                                     'Product Name': 'Electronics - 500',
                                     'Revenue': '500'}}],
             'returned_rows': 5,
             'role': 'revenue management product data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB, quicksum

def build_and_solve_merchandise_allocation(CSVQA_DATA):
    table_id = 'file_0_view_0'
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    if table_id not in tables:
        raise RuntimeError(f'Required table_id {table_id} not found in CSVQA_DATA.')
    table = tables[table_id]
    records = table['records']
    I = []
    r = {}
    d = {}
    s = {}
    for rec in records:
        vals = rec['values']
        prod = vals['Product Name']
        try:
            revenue = float(vals['Revenue'])
            demand = float(vals['Demand'])
            inventory = float(vals['Initial Inventory'])
        except Exception as e:
            raise RuntimeError(f'Non-numeric data in record {prod}: {e}')
        I.append(prod)
        r[prod] = revenue
        d[prod] = demand
        s[prod] = inventory
    if not set(I) == set(r) == set(d) == set(s):
        raise RuntimeError('Mismatch in index sets or missing parameter data.')
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    demand_constrs = m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
    inventory_constrs = m.addConstrs((x_vars[i] <= s[i] for i in I), name='')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve_merchandise_allocation(CSVQA_DATA)