CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 19,
             'returned_rows': 19,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': '20in Monitor',
                                     'Revenue': '38.4965',
                                     'Demand': '8230',
                                     'Initial Inventory': '41290'}},
                         {'source_row': 1,
                          'values': {'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '261.2933',
                                     'Demand': '12474',
                                     'Initial Inventory': '62440'}},
                         {'source_row': 2,
                          'values': {'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965',
                                     'Demand': '15057',
                                     'Initial Inventory': '75500'}},
                         {'source_row': 3,
                          'values': {'Product Name': '34in Ultrawide Monitor',
                                     'Revenue': '254.5933',
                                     'Demand': '12380',
                                     'Initial Inventory': '61990'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'AA Batteries (4-pack)',
                                     'Revenue': '1.92',
                                     'Demand': '49137',
                                     'Initial Inventory': '276350'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'AAA Batteries (4-pack)',
                                     'Revenue': '1.495',
                                     'Demand': '53353',
                                     'Initial Inventory': '310170'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Apple Airpods Headphones',
                                     'Revenue': '52.5',
                                     'Demand': '31211',
                                     'Initial Inventory': '156610'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'Bose SoundSport Headphones',
                                     'Revenue': '49.995',
                                     'Demand': '26782',
                                     'Initial Inventory': '134570'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'Flatscreen TV',
                                     'Revenue': '201.0',
                                     'Demand': '9619',
                                     'Initial Inventory': '48190'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'Google Phone',
                                     'Revenue': '402.0',
                                     'Demand': '11057',
                                     'Initial Inventory': '55320'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'LG Dryer',
                                     'Revenue': '402.0',
                                     'Demand': '1292',
                                     'Initial Inventory': '6460'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'LG Washing Machine',
                                     'Revenue': '402.0',
                                     'Demand': '1332',
                                     'Initial Inventory': '6660'}},
                         {'source_row': 12,
                          'values': {'Product Name': 'Lightning Charging Cable',
                                     'Revenue': '7.475',
                                     'Demand': '44932',
                                     'Initial Inventory': '232170'}},
                         {'source_row': 13,
                          'values': {'Product Name': 'Macbook Pro Laptop',
                                     'Revenue': '1139.0',
                                     'Demand': '9452',
                                     'Initial Inventory': '47280'}},
                         {'source_row': 14,
                          'values': {'Product Name': 'ThinkPad Laptop',
                                     'Revenue': '669.9933',
                                     'Demand': '8258',
                                     'Initial Inventory': '41300'}},
                         {'source_row': 15,
                          'values': {'Product Name': 'USB-C Charging Cable',
                                     'Revenue': '5.975',
                                     'Demand': '45987',
                                     'Initial Inventory': '239750'}},
                         {'source_row': 16,
                          'values': {'Product Name': 'Vareebadd Phone',
                                     'Revenue': '268.0',
                                     'Demand': '4133',
                                     'Initial Inventory': '20680'}},
                         {'source_row': 17,
                          'values': {'Product Name': 'Wired Headphones',
                                     'Revenue': '11.99',
                                     'Demand': '39525',
                                     'Initial Inventory': '205570'}},
                         {'source_row': 18,
                          'values': {'Product Name': 'iPhone',
                                     'Revenue': '469.0',
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
if not records:
    raise ValueError('No records found in file_0_view_0.')
product_indices = []
product_names = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    pname = rec['values']['Product Name']
    if not isinstance(pname, str):
        continue
    if re.search('27in', pname):
        product_indices.append(pname)
        product_names.append(pname)
        try:
            rev = rec['values']['Revenue']
            dem = rec['values']['Demand']
            inv = rec['values']['Initial Inventory']
        except KeyError as e:
            raise ValueError(f'Missing required column: {e}')
        try:
            revenue[pname] = float(rev)
            demand[pname] = int(float(dem))
            inventory[pname] = int(float(inv))
        except Exception as e:
            raise ValueError(f"Error parsing numeric fields for product '{pname}': {e}")
for pname in product_indices:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing data for product '{pname}'.")
if not product_indices:
    raise ValueError("No products with '27in' in Product Name found.")
m = gp.Model('27in_Product_Revenue_Maximization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_indices, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_indices)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in product_indices), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in product_indices), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')