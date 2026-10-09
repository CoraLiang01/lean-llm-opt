CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 0,
                          'values': {'Demand': '8230',
                                     'Initial Inventory': '41290',
                                     'Product Name': '20in Monitor',
                                     'Revenue': '38.4965'}},
                         {'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '261.2933'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965'}},
                         {'source_row': 3,
                          'values': {'Demand': '12380',
                                     'Initial Inventory': '61990',
                                     'Product Name': '34in Ultrawide Monitor',
                                     'Revenue': '254.5933'}},
                         {'source_row': 4,
                          'values': {'Demand': '49137',
                                     'Initial Inventory': '276350',
                                     'Product Name': 'AA Batteries (4-pack)',
                                     'Revenue': '1.92'}},
                         {'source_row': 5,
                          'values': {'Demand': '53353',
                                     'Initial Inventory': '310170',
                                     'Product Name': 'AAA Batteries (4-pack)',
                                     'Revenue': '1.495'}},
                         {'source_row': 6,
                          'values': {'Demand': '31211',
                                     'Initial Inventory': '156610',
                                     'Product Name': 'Apple Airpods Headphones',
                                     'Revenue': '52.5'}},
                         {'source_row': 7,
                          'values': {'Demand': '26782',
                                     'Initial Inventory': '134570',
                                     'Product Name': 'Bose SoundSport Headphones',
                                     'Revenue': '49.995'}},
                         {'source_row': 8,
                          'values': {'Demand': '9619',
                                     'Initial Inventory': '48190',
                                     'Product Name': 'Flatscreen TV',
                                     'Revenue': '201.0'}},
                         {'source_row': 9,
                          'values': {'Demand': '11057',
                                     'Initial Inventory': '55320',
                                     'Product Name': 'Google Phone',
                                     'Revenue': '402.0'}},
                         {'source_row': 10,
                          'values': {'Demand': '1292',
                                     'Initial Inventory': '6460',
                                     'Product Name': 'LG Dryer',
                                     'Revenue': '402.0'}},
                         {'source_row': 11,
                          'values': {'Demand': '1332',
                                     'Initial Inventory': '6660',
                                     'Product Name': 'LG Washing Machine',
                                     'Revenue': '402.0'}},
                         {'source_row': 12,
                          'values': {'Demand': '44932',
                                     'Initial Inventory': '232170',
                                     'Product Name': 'Lightning Charging Cable',
                                     'Revenue': '7.475'}},
                         {'source_row': 13,
                          'values': {'Demand': '9452',
                                     'Initial Inventory': '47280',
                                     'Product Name': 'Macbook Pro Laptop',
                                     'Revenue': '1139.0'}},
                         {'source_row': 14,
                          'values': {'Demand': '8258',
                                     'Initial Inventory': '41300',
                                     'Product Name': 'ThinkPad Laptop',
                                     'Revenue': '669.9933'}},
                         {'source_row': 15,
                          'values': {'Demand': '45987',
                                     'Initial Inventory': '239750',
                                     'Product Name': 'USB-C Charging Cable',
                                     'Revenue': '5.975'}},
                         {'source_row': 16,
                          'values': {'Demand': '4133',
                                     'Initial Inventory': '20680',
                                     'Product Name': 'Vareebadd Phone',
                                     'Revenue': '268.0'}},
                         {'source_row': 17,
                          'values': {'Demand': '39525',
                                     'Initial Inventory': '205570',
                                     'Product Name': 'Wired Headphones',
                                     'Revenue': '11.99'}},
                         {'source_row': 18,
                          'values': {'Demand': '13691',
                                     'Initial Inventory': '68490',
                                     'Product Name': 'iPhone',
                                     'Revenue': '469.0'}}],
             'returned_rows': 19,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': 'expected a string or tuple, not list',
                'planner_errors': ['expected a string or tuple, not list'],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
    I = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in table['records']:
        pname = rec['values']['Product Name']
        if '27in' in pname.casefold():
            I.append(pname)
            try:
                revenue[pname] = float(rec['values']['Revenue'])
                demand[pname] = int(rec['values']['Demand'])
                inventory[pname] = int(rec['values']['Initial Inventory'])
            except Exception as e:
                raise ValueError(f"Invalid data for product '{pname}': {e}")
    if not I:
        raise ValueError("No products with '27in' in the name found.")
    for pname in I:
        if pname not in revenue or pname not in demand or pname not in inventory:
            raise ValueError(f"Missing data for product '{pname}'.")
    ub = {i: min(demand[i], inventory[i]) for i in I}
    m = gp.Model('27in_Promo')
    x = m.addVars(I, lb=0, ub=ub, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in I), name='')
    m.addConstrs((x[i] <= inventory[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_DATA)