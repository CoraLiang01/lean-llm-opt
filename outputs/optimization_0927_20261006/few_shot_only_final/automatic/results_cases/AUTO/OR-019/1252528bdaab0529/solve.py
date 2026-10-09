CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 19,
             'returned_rows': 19,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': '20in Monitor',
                                     'Revenue': '109.99',
                                     'Demand': '8230',
                                     'Initial Inventory': '41290'}},
                         {'source_row': 1,
                          'values': {'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99',
                                     'Demand': '12474',
                                     'Initial Inventory': '62440'}},
                         {'source_row': 2,
                          'values': {'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99',
                                     'Demand': '15057',
                                     'Initial Inventory': '75500'}},
                         {'source_row': 3,
                          'values': {'Product Name': '34in Ultrawide Monitor',
                                     'Revenue': '379.99',
                                     'Demand': '12380',
                                     'Initial Inventory': '61990'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'AA Batteries (4-pack)',
                                     'Revenue': '3.84',
                                     'Demand': '49129',
                                     'Initial Inventory': '276350'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'AAA Batteries (4-pack)',
                                     'Revenue': '2.99',
                                     'Demand': '53317',
                                     'Initial Inventory': '310170'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Apple Airpods Headphones',
                                     'Revenue': '150.0',
                                     'Demand': '31210',
                                     'Initial Inventory': '156610'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'Bose SoundSport Headphones',
                                     'Revenue': '99.99',
                                     'Demand': '26784',
                                     'Initial Inventory': '134570'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'Flatscreen TV',
                                     'Revenue': '300.0',
                                     'Demand': '9619',
                                     'Initial Inventory': '48190'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'Google Phone',
                                     'Revenue': '600.0',
                                     'Demand': '11057',
                                     'Initial Inventory': '55320'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'LG Dryer',
                                     'Revenue': '600.0',
                                     'Demand': '1292',
                                     'Initial Inventory': '6460'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'LG Washing Machine',
                                     'Revenue': '600.0',
                                     'Demand': '1332',
                                     'Initial Inventory': '6660'}},
                         {'source_row': 12,
                          'values': {'Product Name': 'Lightning Charging Cable',
                                     'Revenue': '14.95',
                                     'Demand': '44936',
                                     'Initial Inventory': '232170'}},
                         {'source_row': 13,
                          'values': {'Product Name': 'Macbook Pro Laptop',
                                     'Revenue': '1700.0',
                                     'Demand': '9452',
                                     'Initial Inventory': '47280'}},
                         {'source_row': 14,
                          'values': {'Product Name': 'ThinkPad Laptop',
                                     'Revenue': '999.99',
                                     'Demand': '8258',
                                     'Initial Inventory': '41300'}},
                         {'source_row': 15,
                          'values': {'Product Name': 'USB-C Charging Cable',
                                     'Revenue': '11.95',
                                     'Demand': '45977',
                                     'Initial Inventory': '239750'}},
                         {'source_row': 16,
                          'values': {'Product Name': 'Vareebadd Phone',
                                     'Revenue': '400.0',
                                     'Demand': '4133',
                                     'Initial Inventory': '20680'}},
                         {'source_row': 17,
                          'values': {'Product Name': 'Wired Headphones',
                                     'Revenue': '11.99',
                                     'Demand': '39520',
                                     'Initial Inventory': '205570'}},
                         {'source_row': 18,
                          'values': {'Product Name': 'iPhone',
                                     'Revenue': '700.0',
                                     'Demand': '13691',
                                     'Initial Inventory': '68490'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import re
import gurobipy as gp
from gurobipy import GRB
tables = CSVQA_DATA['tables']
table = None
for t in tables:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError('Table file_0_view_0 not found in CSVQA_DATA.')
records = table['records']
if records is None:
    raise ValueError('No records found in file_0_view_0.')
product_pattern = re.compile('27in', re.IGNORECASE)
products_27in = []
for rec in records:
    pname = rec['values'].get('Product Name', '')
    if product_pattern.search(pname):
        products_27in.append(pname)
if not products_27in:
    raise ValueError("No products with '27in' in Product Name found.")
revenue = {}
demand = {}
inventory = {}
for rec in records:
    pname = rec['values'].get('Product Name', '')
    if pname in products_27in:
        try:
            rev = float(rec['values']['Revenue'])
            dem = float(rec['values']['Demand'])
            inv = float(rec['values']['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Non-numeric or missing data for product '{pname}': {e}")
        revenue[pname] = rev
        demand[pname] = dem
        inventory[pname] = inv
for pname in products_27in:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing data for product '{pname}'.")
m = gp.Model('27in_Product_Revenue_Maximization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products_27in, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products_27in)), GRB.MAXIMIZE)
for i in products_27in:
    upper_bound = min(demand[i], inventory[i])
    m.addConstr(x_vars[i] <= upper_bound, name=f'ub_{i}')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')