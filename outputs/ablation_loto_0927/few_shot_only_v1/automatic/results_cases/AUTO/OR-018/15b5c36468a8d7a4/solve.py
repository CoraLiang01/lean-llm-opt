CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'Salesdata.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 12,
             'returned_rows': 12,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28',
                                     'Demand': '3066513',
                                     'Initial Inventory': '22749210'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Beverages_47.45',
                                     'Revenue': '47.45',
                                     'Demand': '2961484',
                                     'Initial Inventory': '22049510'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Cereal_205.7',
                                     'Revenue': '205.7',
                                     'Demand': '2621950',
                                     'Initial Inventory': '19459680'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Clothes_109.28',
                                     'Revenue': '109.28',
                                     'Demand': '2660974',
                                     'Initial Inventory': '19754410'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Cosmetics_437.2',
                                     'Revenue': '437.2',
                                     'Demand': '2896197',
                                     'Initial Inventory': '21366410'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Fruits_9.33',
                                     'Revenue': '9.33',
                                     'Demand': '3169426',
                                     'Initial Inventory': '23410830'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Household_668.27',
                                     'Revenue': '668.27',
                                     'Demand': '2846953',
                                     'Initial Inventory': '20986130'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'Meat_421.89',
                                     'Revenue': '421.89',
                                     'Demand': '2546972',
                                     'Initial Inventory': '19011970'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'Office Supplies_651.21',
                                     'Revenue': '651.21',
                                     'Demand': '2855686',
                                     'Initial Inventory': '21062780'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'Personal Care_81.73',
                                     'Revenue': '81.73',
                                     'Demand': '2855360',
                                     'Initial Inventory': '21265920'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'Snacks_152.58',
                                     'Revenue': '152.58',
                                     'Demand': '2592261',
                                     'Initial Inventory': '19155280'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'Vegetables_154.06',
                                     'Revenue': '154.06',
                                     'Demand': '2826603',
                                     'Initial Inventory': '20867490'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import gurobipy as gp
from gurobipy import GRB
import re
tables = CSVQA_DATA['tables']
table = None
for t in tables:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table['records']
if not records:
    raise ValueError("No records found in table 'file_0_view_0'.")
baby_items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    pname = rec['values'].get('Product Name', '')
    if not isinstance(pname, str):
        continue
    if re.search('\\bBaby\\b', pname, re.IGNORECASE):
        baby_items.append(pname)
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
            raise ValueError(f"Invalid data for product '{pname}': {e}")
if not baby_items:
    raise ValueError("No products classified as 'Baby' found in the data.")
for i in baby_items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f"Missing parameter(s) for product '{i}'.")
m = gp.Model('Baby_Product_Revenue_Max')
x = m.addVars(baby_items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in baby_items)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in baby_items), name='')
m.addConstrs((x[i] <= demand[i] for i in baby_items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')