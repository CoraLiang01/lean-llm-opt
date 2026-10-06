CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail store operates a complex supply chain to distribute products across various categories, serving '
          'customers in multiple geographic regions. The daily demand of customer groups is recorded in '
          '"customer_demand.csv", while the supply capacity of distribution centers is detailed in '
          '"supply_capacity.csv". The transportation cost per unit of goods from each distribution center to each '
          'customer group is recorded in "transportation_costs.csv". The objective is to develop an optimal '
          'fulfillment policy that determines how goods should be allocated from distribution centers to customer '
          'groups, ensuring that all demands are met without exceeding supply capacities, while minimizing the total '
          'transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer_id', 'table_id': 'file_0_view_0'},
                    'column_id_mapping': {'transportation_cost_to_C1': 'C1',
                                          'transportation_cost_to_C10': 'C10',
                                          'transportation_cost_to_C11': 'C11',
                                          'transportation_cost_to_C12': 'C12',
                                          'transportation_cost_to_C2': 'C2',
                                          'transportation_cost_to_C3': 'C3',
                                          'transportation_cost_to_C4': 'C4',
                                          'transportation_cost_to_C5': 'C5',
                                          'transportation_cost_to_C6': 'C6',
                                          'transportation_cost_to_C7': 'C7',
                                          'transportation_cost_to_C8': 'C8',
                                          'transportation_cost_to_C9': 'C9'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'supplier_id', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'supplier_id',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer_id', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0, 'values': {'customer_id': 'C1', 'demand': '52'}},
                         {'source_row': 1, 'values': {'customer_id': 'C2', 'demand': '80'}},
                         {'source_row': 2, 'values': {'customer_id': 'C3', 'demand': '392'}},
                         {'source_row': 3, 'values': {'customer_id': 'C4', 'demand': '103'}},
                         {'source_row': 4, 'values': {'customer_id': 'C5', 'demand': '32'}},
                         {'source_row': 5, 'values': {'customer_id': 'C6', 'demand': '1426'}},
                         {'source_row': 6, 'values': {'customer_id': 'C7', 'demand': '1024'}},
                         {'source_row': 7, 'values': {'customer_id': 'C8', 'demand': '2736'}},
                         {'source_row': 8, 'values': {'customer_id': 'C9', 'demand': '1129'}},
                         {'source_row': 9, 'values': {'customer_id': 'C10', 'demand': '676'}},
                         {'source_row': 10, 'values': {'customer_id': 'C11', 'demand': '2631'}},
                         {'source_row': 11, 'values': {'customer_id': 'C12', 'demand': '31'}}],
             'returned_rows': 12,
             'role': 'customer demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['supplier_id', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0, 'values': {'supplier_id': 'S1', 'supply_capacity': '58'}},
                         {'source_row': 1, 'values': {'supplier_id': 'S2', 'supply_capacity': '32'}},
                         {'source_row': 2, 'values': {'supplier_id': 'S3', 'supply_capacity': '6161'}},
                         {'source_row': 3, 'values': {'supplier_id': 'S4', 'supply_capacity': '4'}},
                         {'source_row': 4, 'values': {'supplier_id': 'S5', 'supply_capacity': '47'}},
                         {'source_row': 5, 'values': {'supplier_id': 'S6', 'supply_capacity': '178'}},
                         {'source_row': 6, 'values': {'supplier_id': 'S7', 'supply_capacity': '142'}},
                         {'source_row': 7, 'values': {'supplier_id': 'S8', 'supply_capacity': '164'}},
                         {'source_row': 8, 'values': {'supplier_id': 'S9', 'supply_capacity': '1011'}},
                         {'source_row': 9, 'values': {'supplier_id': 'S10', 'supply_capacity': '6'}},
                         {'source_row': 10, 'values': {'supplier_id': 'S11', 'supply_capacity': '7081'}},
                         {'source_row': 11, 'values': {'supplier_id': 'S12', 'supply_capacity': '948'}}],
             'returned_rows': 12,
             'role': 'supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['supplier_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'transportation_cost_to_C4',
                         'transportation_cost_to_C5',
                         'transportation_cost_to_C6',
                         'transportation_cost_to_C7',
                         'transportation_cost_to_C8',
                         'transportation_cost_to_C9',
                         'transportation_cost_to_C10',
                         'transportation_cost_to_C11',
                         'transportation_cost_to_C12'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'supplier_id': 'S1',
                                     'transportation_cost_to_C1': '134.72882437877243',
                                     'transportation_cost_to_C10': '82.52035827662691',
                                     'transportation_cost_to_C11': '1538.830198350349',
                                     'transportation_cost_to_C12': '2320.1146477101956',
                                     'transportation_cost_to_C2': '72.30045141110347',
                                     'transportation_cost_to_C3': '37.9759927355347',
                                     'transportation_cost_to_C4': '611.84656497826',
                                     'transportation_cost_to_C5': '1650.3353902326157',
                                     'transportation_cost_to_C6': '32.97044486689555',
                                     'transportation_cost_to_C7': '34.99837373425947',
                                     'transportation_cost_to_C8': '73.25207370031997',
                                     'transportation_cost_to_C9': '165.752089835103'}},
                         {'source_row': 1,
                          'values': {'supplier_id': 'S2',
                                     'transportation_cost_to_C1': '23.835369186583733',
                                     'transportation_cost_to_C10': '82.8115250443967',
                                     'transportation_cost_to_C11': '1702.1724763823142',
                                     'transportation_cost_to_C12': '1925.8446970545492',
                                     'transportation_cost_to_C2': '1128.6332865368709',
                                     'transportation_cost_to_C3': '187.05459590556023',
                                     'transportation_cost_to_C4': '1.6077337129635412',
                                     'transportation_cost_to_C5': '2227.7036991312493',
                                     'transportation_cost_to_C6': '72.44693148708649',
                                     'transportation_cost_to_C7': '12.60074797346823',
                                     'transportation_cost_to_C8': '1078.7729548111274',
                                     'transportation_cost_to_C9': '383.7220569377865'}},
                         {'source_row': 2,
                          'values': {'supplier_id': 'S3',
                                     'transportation_cost_to_C1': '1138.498759622659',
                                     'transportation_cost_to_C10': '664.1846792537777',
                                     'transportation_cost_to_C11': '36.70238978444623',
                                     'transportation_cost_to_C12': '1405.0506919253128',
                                     'transportation_cost_to_C2': '231.8983422482863',
                                     'transportation_cost_to_C3': '44.1568226726957',
                                     'transportation_cost_to_C4': '962.5481640749796',
                                     'transportation_cost_to_C5': '980.1306032437444',
                                     'transportation_cost_to_C6': '1107.5157506568996',
                                     'transportation_cost_to_C7': '741.7073424442746',
                                     'transportation_cost_to_C8': '136.70204401709762',
                                     'transportation_cost_to_C9': '1182.5161115443857'}},
                         {'source_row': 3,
                          'values': {'supplier_id': 'S4',
                                     'transportation_cost_to_C1': '1043.7562803117191',
                                     'transportation_cost_to_C10': '642.5520451224766',
                                     'transportation_cost_to_C11': '437.9556823253786',
                                     'transportation_cost_to_C12': '76.12498668603241',
                                     'transportation_cost_to_C2': '24.207016063238527',
                                     'transportation_cost_to_C3': '1120.9890438994094',
                                     'transportation_cost_to_C4': '1027.5437638768522',
                                     'transportation_cost_to_C5': '893.4457577903337',
                                     'transportation_cost_to_C6': '1244.2626779205734',
                                     'transportation_cost_to_C7': '980.0297707948428',
                                     'transportation_cost_to_C8': '513.5650272201696',
                                     'transportation_cost_to_C9': '977.6321548115833'}},
                         {'source_row': 4,
                          'values': {'supplier_id': 'S5',
                                     'transportation_cost_to_C1': '4.939930661779214',
                                     'transportation_cost_to_C10': '1783.0865677438942',
                                     'transportation_cost_to_C11': '1619.6154655942953',
                                     'transportation_cost_to_C12': '116.08497905117572',
                                     'transportation_cost_to_C2': '1278.1815735383598',
                                     'transportation_cost_to_C3': '549.7570743190075',
                                     'transportation_cost_to_C4': '21.335474111511868',
                                     'transportation_cost_to_C5': '98.32689279702352',
                                     'transportation_cost_to_C6': '452.0903462786745',
                                     'transportation_cost_to_C7': '595.5981933612871',
                                     'transportation_cost_to_C8': '70.84029746549007',
                                     'transportation_cost_to_C9': '0.001236242263106387'}},
                         {'source_row': 5,
                          'values': {'supplier_id': 'S6',
                                     'transportation_cost_to_C1': '2105.5398596409113',
                                     'transportation_cost_to_C10': '788.4640306393768',
                                     'transportation_cost_to_C11': '818.9532260267922',
                                     'transportation_cost_to_C12': '17.249645794282493',
                                     'transportation_cost_to_C2': '1340.9770559580859',
                                     'transportation_cost_to_C3': '2077.2747284488696',
                                     'transportation_cost_to_C4': '2202.2515396165613',
                                     'transportation_cost_to_C5': '20.5336958780729',
                                     'transportation_cost_to_C6': '2514.0983598337716',
                                     'transportation_cost_to_C7': '2393.213466538681',
                                     'transportation_cost_to_C8': '1197.765543404598',
                                     'transportation_cost_to_C9': '102.32440248566687'}},
                         {'source_row': 6,
                          'values': {'supplier_id': 'S7',
                                     'transportation_cost_to_C1': '61.36183063260923',
                                     'transportation_cost_to_C10': '519.3858364273143',
                                     'transportation_cost_to_C11': '483.3345987058458',
                                     'transportation_cost_to_C12': '1412.7652882009972',
                                     'transportation_cost_to_C2': '113.23716964851326',
                                     'transportation_cost_to_C3': '50.231076475261865',
                                     'transportation_cost_to_C4': '1219.1951077346878',
                                     'transportation_cost_to_C5': '869.868986439361',
                                     'transportation_cost_to_C6': '58.3717723017115',
                                     'transportation_cost_to_C7': '957.3213131437595',
                                     'transportation_cost_to_C8': '168.9964798230943',
                                     'transportation_cost_to_C9': '1363.342713601386'}},
                         {'source_row': 7,
                          'values': {'supplier_id': 'S8',
                                     'transportation_cost_to_C1': '1169.3640988900754',
                                     'transportation_cost_to_C10': '72.78392607232098',
                                     'transportation_cost_to_C11': '1479.1137195486635',
                                     'transportation_cost_to_C12': '65.87463475119084',
                                     'transportation_cost_to_C2': '1037.2170744709088',
                                     'transportation_cost_to_C3': '732.3577706907058',
                                     'transportation_cost_to_C4': '865.2255769415611',
                                     'transportation_cost_to_C5': '1510.0296187049466',
                                     'transportation_cost_to_C6': '780.853789775776',
                                     'transportation_cost_to_C7': '860.7971696230202',
                                     'transportation_cost_to_C8': '935.9126432821531',
                                     'transportation_cost_to_C9': '61.82504909829045'}},
                         {'source_row': 8,
                          'values': {'supplier_id': 'S9',
                                     'transportation_cost_to_C1': '7.936521667214095',
                                     'transportation_cost_to_C10': '1580.7262556456867',
                                     'transportation_cost_to_C11': '1423.5615412405277',
                                     'transportation_cost_to_C12': '2000.5637996766272',
                                     'transportation_cost_to_C2': '1357.5387610434743',
                                     'transportation_cost_to_C3': '628.3001825422914',
                                     'transportation_cost_to_C4': '25.245141819018265',
                                     'transportation_cost_to_C5': '1760.0085249934855',
                                     'transportation_cost_to_C6': '604.6583535734275',
                                     'transportation_cost_to_C7': '696.677483277236',
                                     'transportation_cost_to_C8': '1586.4093183806565',
                                     'transportation_cost_to_C9': '89.68006942976479'}},
                         {'source_row': 9,
                          'values': {'supplier_id': 'S10',
                                     'transportation_cost_to_C1': '1685.3409758586952',
                                     'transportation_cost_to_C10': '2.676413573774264',
                                     'transportation_cost_to_C11': '202.72671236450077',
                                     'transportation_cost_to_C12': '45.95308313010429',
                                     'transportation_cost_to_C2': '437.39384912973503',
                                     'transportation_cost_to_C3': '1568.8939936485654',
                                     'transportation_cost_to_C4': '1486.9268967582923',
                                     'transportation_cost_to_C5': '498.37544466423503',
                                     'transportation_cost_to_C6': '1493.3860214089566',
                                     'transportation_cost_to_C7': '70.17952878640685',
                                     'transportation_cost_to_C8': '526.999992252258',
                                     'transportation_cost_to_C9': '1527.6606596831157'}},
                         {'source_row': 10,
                          'values': {'supplier_id': 'S11',
                                     'transportation_cost_to_C1': '937.1094078301182',
                                     'transportation_cost_to_C10': '1274.3227526636908',
                                     'transportation_cost_to_C11': '1574.7961500081667',
                                     'transportation_cost_to_C12': '1826.428257280374',
                                     'transportation_cost_to_C2': '903.0141415848653',
                                     'transportation_cost_to_C3': '264.7208998642393',
                                     'transportation_cost_to_C4': '21.823340175910644',
                                     'transportation_cost_to_C5': '1661.105486348982',
                                     'transportation_cost_to_C6': '18.59349093795307',
                                     'transportation_cost_to_C7': '372.0817876391536',
                                     'transportation_cost_to_C8': '956.6955598743717',
                                     'transportation_cost_to_C9': '42.9120185256587'}},
                         {'source_row': 11,
                          'values': {'supplier_id': 'S12',
                                     'transportation_cost_to_C1': '1685.7576850724427',
                                     'transportation_cost_to_C10': '0.010486218015015225',
                                     'transportation_cost_to_C11': '11.1693204455916',
                                     'transportation_cost_to_C12': '963.9941814871104',
                                     'transportation_cost_to_C2': '377.3256672852414',
                                     'transportation_cost_to_C3': '1347.0162725120726',
                                     'transportation_cost_to_C4': '1737.020030508406',
                                     'transportation_cost_to_C5': '23.612177980608287',
                                     'transportation_cost_to_C6': '83.08521829288628',
                                     'transportation_cost_to_C7': '1476.1258241571823',
                                     'transportation_cost_to_C8': '530.0021927141197',
                                     'transportation_cost_to_C9': '1782.8442633270377'}}],
             'returned_rows': 12,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'unique_complete_suffix',
                                   'expected_shape': [12, 12],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [12, 12]}],
                'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    customer_demand_table = None
    supply_capacity_table = None
    transportation_costs_table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            customer_demand_table = t
        elif t['table_id'] == 'file_1_view_0':
            supply_capacity_table = t
        elif t['table_id'] == 'file_2_view_0':
            transportation_costs_table = t
    if customer_demand_table is None or supply_capacity_table is None or transportation_costs_table is None:
        raise RuntimeError('Missing required data tables.')
    suppliers = [rec['values']['supplier_id'] for rec in supply_capacity_table['records']]
    customers = [rec['values']['customer_id'] for rec in customer_demand_table['records']]
    demand = {}
    for rec in customer_demand_table['records']:
        cid = rec['values']['customer_id']
        demand[cid] = float(rec['values']['demand'])
    supply_capacity = {}
    for rec in supply_capacity_table['records']:
        sid = rec['values']['supplier_id']
        supply_capacity[sid] = float(rec['values']['supply_capacity'])
    col_mapping = None
    for rel in CSVQA_DATA.get('relationships', []):
        if rel.get('matrix_table_id') == 'file_2_view_0':
            col_mapping = rel.get('column_id_mapping')
            break
    if col_mapping is None:
        raise RuntimeError('Missing column mapping for transportation cost matrix.')
    cost = {}
    for rec in transportation_costs_table['records']:
        sid = rec['values']['supplier_id']
        cost[sid] = {}
        for (col, cid) in col_mapping.items():
            cost[sid][cid] = float(rec['values'][col])
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in suppliers:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for supplier {i}, customer {j}')
    m = gp.Model('Original_RAG_TP')
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()