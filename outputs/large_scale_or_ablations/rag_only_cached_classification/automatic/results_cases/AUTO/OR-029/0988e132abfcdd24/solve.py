CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'An e-commerce platform sells women’s clothing items with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘FAUX’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent '
          'the number of units of each ‘FAUX’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'ZARASales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'FAUX' product i",
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'FAUX'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'FAUX' product i",
                                         'inclusive': 'both',
                                         'operator': 'contains',
                                         'value': 'FAUX'}],
                         'logic': 'or'},
             'original_rows': 194,
             'records': [{'source_row': 56,
                          'values': {'Demand': '3025',
                                     'Initial Inventory': '20970',
                                     'Product Name': 'FAUX FUR JEWEL SWEATER',
                                     'Revenue': '35.9'}},
                         {'source_row': 57,
                          'values': {'Demand': '9585',
                                     'Initial Inventory': '71970',
                                     'Product Name': 'FAUX LEATHER BOMBER JACKET',
                                     'Revenue': '69.9'}},
                         {'source_row': 58,
                          'values': {'Demand': '4486',
                                     'Initial Inventory': '32730',
                                     'Product Name': 'FAUX LEATHER BOXY FIT JACKET',
                                     'Revenue': '99.9'}},
                         {'source_row': 59,
                          'values': {'Demand': '10322',
                                     'Initial Inventory': '71130',
                                     'Product Name': 'FAUX LEATHER JACKET',
                                     'Revenue': '99.9'}},
                         {'source_row': 60,
                          'values': {'Demand': '4868',
                                     'Initial Inventory': '34910',
                                     'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION',
                                     'Revenue': '159.0'}},
                         {'source_row': 61,
                          'values': {'Demand': '8482',
                                     'Initial Inventory': '64010',
                                     'Product Name': 'FAUX LEATHER PUFFER JACKET',
                                     'Revenue': '69.99'}},
                         {'source_row': 62,
                          'values': {'Demand': '2607',
                                     'Initial Inventory': '20760',
                                     'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS',
                                     'Revenue': '99.9'}},
                         {'source_row': 63,
                          'values': {'Demand': '1784',
                                     'Initial Inventory': '12490',
                                     'Product Name': 'FAUX SHEARLING PLAID JACKET',
                                     'Revenue': '89.9'}},
                         {'source_row': 64,
                          'values': {'Demand': '6626',
                                     'Initial Inventory': '50300',
                                     'Product Name': 'FAUX SUEDE BOMBER JACKET',
                                     'Revenue': '69.9'}},
                         {'source_row': 65,
                          'values': {'Demand': '3256',
                                     'Initial Inventory': '24570',
                                     'Product Name': 'FAUX SUEDE JACKET',
                                     'Revenue': '89.9'}},
                         {'source_row': 66,
                          'values': {'Demand': '2955',
                                     'Initial Inventory': '24430',
                                     'Product Name': 'FAUX SUEDE OVERSHIRT',
                                     'Revenue': '69.9'}},
                         {'source_row': 67,
                          'values': {'Demand': '910',
                                     'Initial Inventory': '7070',
                                     'Product Name': 'FAUX SUEDE PATCH JACKET',
                                     'Revenue': '89.9'}}],
             'returned_rows': 12,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB

def solve_faux_optimization(CSVQA_DATA):
    faux_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            faux_table = t
            break
    if faux_table is None:
        raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
    faux_products = []
    faux_pattern = re.compile('FAUX', re.IGNORECASE)
    for rec in faux_table['records']:
        pname = rec['values']['Product Name']
        if faux_pattern.search(pname):
            faux_products.append(pname)
    seen = set()
    faux_products = [x for x in faux_products if not (x in seen or seen.add(x))]
    product_records = {rec['values']['Product Name']: rec for rec in faux_table['records']}
    r = {}
    d = {}
    s = {}
    for pname in faux_products:
        rec = product_records.get(pname)
        if rec is None:
            raise RuntimeError(f"Missing record for product '{pname}' in table_id 'file_0_view_0'.")
        try:
            r[pname] = float(rec['values']['Revenue'])
            d[pname] = int(rec['values']['Demand'])
            s[pname] = int(rec['values']['Initial Inventory'])
        except Exception as e:
            raise RuntimeError(f"Error parsing parameters for product '{pname}': {e}")
    if set(faux_products) != set(r.keys()) or set(faux_products) != set(d.keys()) or set(faux_products) != set(s.keys()):
        raise RuntimeError('Mismatch in index set and parameter coverage for products.')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(faux_products, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((r[i] * x[i] for i in faux_products)), GRB.MAXIMIZE)
    for i in faux_products:
        m.addConstr(x[i] <= s[i], name='')
        m.addConstr(x[i] <= d[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for i in faux_products:
            print(x[i].VarName, x[i].X)
    else:
        print(m.Status)
    return m
m = solve_faux_optimization(CSVQA_DATA)