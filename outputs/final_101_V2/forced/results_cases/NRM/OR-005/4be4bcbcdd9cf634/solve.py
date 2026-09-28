CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Amazon runs several distribution centers that deliver essential goods daily to different customer groups. '
          'The daily demand for each customer group is listed in “customer_demand.csv,” while the daily supply '
          'capacity of the distribution centers is outlined in “supply_capacity.csv.” The transportation cost per unit '
          'from each distribution center to each customer group is specified in “transportation_costs.csv.” The goal '
          'is to decide the quantity of goods to be shipped from each distribution center to each customer group, '
          'ensuring all demands are fulfilled without exceeding the supply capacity of any center, while minimizing '
          'the total transportation cost.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Customers', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Customers': 'demand1', 'demand': '9'}},
                         {'source_row': 1, 'values': {'Customers': 'demand2', 'demand': '66'}},
                         {'source_row': 2, 'values': {'Customers': 'demand3', 'demand': '56'}},
                         {'source_row': 3, 'values': {'Customers': 'demand4', 'demand': '17'}},
                         {'source_row': 4, 'values': {'Customers': 'demand5', 'demand': '43'}},
                         {'source_row': 5, 'values': {'Customers': 'demand6', 'demand': '62'}},
                         {'source_row': 6, 'values': {'Customers': 'demand7', 'demand': '10'}},
                         {'source_row': 7, 'values': {'Customers': 'demand8', 'demand': '37'}}],
             'returned_rows': 8,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['Supplier', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Supplier': 'supplier1', 'supply_capacity': '60'}},
                         {'source_row': 1, 'values': {'Supplier': 'supplier2', 'supply_capacity': '22'}},
                         {'source_row': 2, 'values': {'Supplier': 'supplier3', 'supply_capacity': '16'}},
                         {'source_row': 3, 'values': {'Supplier': 'supplier4', 'supply_capacity': '14'}},
                         {'source_row': 4, 'values': {'Supplier': 'supplier5', 'supply_capacity': '19'}},
                         {'source_row': 5, 'values': {'Supplier': 'supplier6', 'supply_capacity': '70'}},
                         {'source_row': 6, 'values': {'Supplier': 'supplier7', 'supply_capacity': '60'}},
                         {'source_row': 7, 'values': {'Supplier': 'supplier8', 'supply_capacity': '39'}}],
             'returned_rows': 8,
             'role': 'file_1',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0',
                         'demand1',
                         'demand2',
                         'demand3',
                         'demand4',
                         'demand5',
                         'demand6',
                         'demand7',
                         'demand8'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 0': 'supply1',
                                     'demand1': '0.03020736643461065',
                                     'demand2': '229.50723504640203',
                                     'demand3': '198.62356558205792',
                                     'demand4': '12.995050640153751',
                                     'demand5': '211.20732124396406',
                                     'demand6': '134.9442985029274',
                                     'demand7': '9.822206398831067',
                                     'demand8': '11.394077543225675'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 0': 'supply2',
                                     'demand1': '232.34691308087835',
                                     'demand2': '3.6258726438627473',
                                     'demand3': '0.28605434149404785',
                                     'demand4': '45.73127693242935',
                                     'demand5': '2.8304796563034573',
                                     'demand6': '107.05891033185472',
                                     'demand7': '299.96317913389305',
                                     'demand8': '23.79935436307657'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 0': 'supply3',
                                     'demand1': '11.061938334356302',
                                     'demand2': '0.2041995326579051',
                                     'demand3': '0.2789447278030927',
                                     'demand4': '45.721912724349636',
                                     'demand5': '59.54895565737313',
                                     'demand6': '5.097536739581239',
                                     'demand7': '300.00118415135785',
                                     'demand8': '23.711282707746893'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 0': 'supply4',
                                     'demand1': '235.1794835706472',
                                     'demand2': '43.794668963036194',
                                     'demand3': '40.709846782945924',
                                     'demand4': '0.07774496620087613',
                                     'demand5': '4.237728183419554',
                                     'demand6': '131.70915517494691',
                                     'demand7': '296.55587567706743',
                                     'demand8': '29.810940017561297'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 0': 'supply5',
                                     'demand1': '211.85808746383796',
                                     'demand2': '47.60180876530328',
                                     'demand3': '50.04007716193931',
                                     'demand4': '86.14548807358399',
                                     'demand5': '0.06197897916874956',
                                     'demand6': '5.3345515296262205',
                                     'demand7': '270.06290423798396',
                                     'demand8': '3.853933133973331'}},
                         {'source_row': 5,
                          'values': {'Unnamed: 0': 'supply6',
                                     'demand1': '6.45506633554524',
                                     'demand2': '88.16323623354015',
                                     'demand3': '5.047091671641611',
                                     'demand4': '151.46120287365497',
                                     'demand5': '5.290760161059401',
                                     'demand6': '0.04602205335871525',
                                     'demand7': '9.93670660180487',
                                     'demand8': '103.75460989446313'}},
                         {'source_row': 6,
                          'values': {'Unnamed: 0': 'supply7',
                                     'demand1': '174.27229047340035',
                                     'demand2': '250.58223528739327',
                                     'demand3': '253.90413041857263',
                                     'demand4': '16.235467318386764',
                                     'demand5': '12.643140514778086',
                                     'demand6': '175.0672824108511',
                                     'demand7': '2.983839625303656',
                                     'demand8': '317.0655193866389'}},
                         {'source_row': 7,
                          'values': {'Unnamed: 0': 'supply8',
                                     'demand1': '207.87006253790491',
                                     'demand2': '1.517168471518212',
                                     'demand3': '24.027239288137153',
                                     'demand4': '27.133999276450346',
                                     'demand5': '73.20672468851855',
                                     'demand6': '125.72910359893308',
                                     'demand7': '15.463103251642147',
                                     'demand8': '0.20164987511903337'}}],
             'returned_rows': 8,
             'role': 'file_2',
             'table_id': 'file_2_view_0'}],
 'validation': {'fallback_reason': "Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [8, 8], "
                                   "'expected_shape': [8, 8], 'row_ids_aligned': False, 'column_ids_aligned': True}",
                'planner_errors': ["Matrix validation failed: {'matrix_table_id': 'file_2_view_0', 'shape': [8, 8], "
                                   "'expected_shape': [8, 8], 'row_ids_aligned': False, 'column_ids_aligned': True}"],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
data = CSVQA_DATA
supply_table = next((t for t in data['tables'] if t['table_id'] == 'file_1_view_0'))
suppliers = [rec['values']['Supplier'] for rec in supply_table['records']]
a_s = {rec['values']['Supplier']: float(rec['values']['supply_capacity']) for rec in supply_table['records']}
demand_table = next((t for t in data['tables'] if t['table_id'] == 'file_0_view_0'))
customers = [rec['values']['Customers'] for rec in demand_table['records']]
b_d = {rec['values']['Customers']: float(rec['values']['demand']) for rec in demand_table['records']}
cost_table = next((t for t in data['tables'] if t['table_id'] == 'file_2_view_0'))
cost_suppliers = [rec['values']['Unnamed: 0'] for rec in cost_table['records']]
cost_customers = [col for col in cost_table['columns'] if col.startswith('demand')]
c_sd = {}
for rec in cost_table['records']:
    s = rec['values']['Unnamed: 0']
    for d in cost_customers:
        c_sd[s, d] = float(rec['values'][d])
if set(suppliers) != set(cost_suppliers):
    raise ValueError('Mismatch between suppliers in supply_capacity and transportation_costs tables.')
if set(customers) != set(cost_customers):
    raise ValueError('Mismatch between customers in customer_demand and transportation_costs tables.')
for s in suppliers:
    for d in customers:
        if (s, d) not in c_sd:
            raise ValueError(f'Missing transportation cost for ({s}, {d})')
m = gp.Model('Amazon_Distribution_Transportation')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((c_sd[s, d] * x[s, d] for s in suppliers for d in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[s, d] for s in suppliers)) == b_d[d] for d in customers), name='')
m.addConstrs((gp.quicksum((x[s, d] for d in customers)) <= a_s[s] for s in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')