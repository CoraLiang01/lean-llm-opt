CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Walmart stores play a crucial role in delivering essential goods to a range of customer groups daily. The '
          'problem focuses on determining the optimal transportation plan to meet customer demands, with daily demand '
          'data provided in "customer_demand.csv". These demands are fulfilled using supplies from Walmart stores, '
          'each with a specific daily supply capacity detailed in "supply_capacity.csv". The transportation cost per '
          'unit of goods from each Walmart store to each customer group is recorded in "transportation_costs.csv". The '
          'goal is to calculate the quantities to be transported from each Walmart store to each customer group, '
          'ensuring that all demands are satisfied without exceeding the supply capacity of any store, while '
          'minimizing the overall transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '11'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '1148'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '54'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '833'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '154'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '551'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '7081'}},
                         {'source_row': 7, 'values': {'customer': 'C8', 'demand': '76'}},
                         {'source_row': 8, 'values': {'customer': 'C9', 'demand': '66'}},
                         {'source_row': 9, 'values': {'customer': 'C10', 'demand': '174'}},
                         {'source_row': 10, 'values': {'customer': 'C11', 'demand': '15'}},
                         {'source_row': 11, 'values': {'customer': 'C12', 'demand': '680'}}],
             'returned_rows': 12,
             'role': 'customer demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '4'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '575'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '1504'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '178'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'supply_capacity': '228'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'supply_capacity': '50'}},
                         {'source_row': 6, 'values': {'Unnamed: 0': 'S7', 'supply_capacity': '3'}},
                         {'source_row': 7, 'values': {'Unnamed: 0': 'S8', 'supply_capacity': '6148'}},
                         {'source_row': 8, 'values': {'Unnamed: 0': 'S9', 'supply_capacity': '6'}},
                         {'source_row': 9, 'values': {'Unnamed: 0': 'S10', 'supply_capacity': '10673'}},
                         {'source_row': 10, 'values': {'Unnamed: 0': 'S11', 'supply_capacity': '174'}}],
             'returned_rows': 11,
             'role': 'store supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 11,
             'records': [{'source_row': 0,
                          'values': {'C1': '0.639144476970582',
                                     'C10': '52.49127007738756',
                                     'C11': '606.4434399601076',
                                     'C12': '1192.4686514332489',
                                     'C2': '49.71842803015729',
                                     'C3': '33.75857739960576',
                                     'C4': '1570.673110465785',
                                     'C5': '1370.4095474322417',
                                     'C6': '57.35307774277479',
                                     'C7': '57.18299486453194',
                                     'C8': '54.9209612366192',
                                     'C9': '1143.680909226399',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '605.4786373569875',
                                     'C10': '1472.7481944140839',
                                     'C11': '0.6004591535232997',
                                     'C12': '49.86854015671519',
                                     'C2': '64.53562572761275',
                                     'C3': '478.4779031378926',
                                     'C4': '887.0480739088434',
                                     'C5': '65.46111249492031',
                                     'C6': '71.93605217833378',
                                     'C7': '41.29015388498019',
                                     'C8': '70.36038207491039',
                                     'C9': '35.35892996332259',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '1139.0440074582496',
                                     'C10': '162.70556734409897',
                                     'C11': '1208.613484750161',
                                     'C12': '110.18688517926226',
                                     'C2': '4.785056325458736',
                                     'C3': '1805.6214229758102',
                                     'C4': '1302.8958147418275',
                                     'C5': '2437.321229159901',
                                     'C6': '103.80368582531935',
                                     'C7': '774.6558236505713',
                                     'C8': '4.515988277174664',
                                     'C9': '879.7048537066717',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '69.26989890601938',
                                     'C10': '94.6504089308906',
                                     'C11': '1277.251451477508',
                                     'C12': '21.636190287574664',
                                     'C2': '2105.485387219297',
                                     'C3': '869.6820232492624',
                                     'C4': '1494.8985656180187',
                                     'C5': '310.5376623181487',
                                     'C6': '98.15455717980421',
                                     'C7': '103.36918486373995',
                                     'C8': '1758.8783768888936',
                                     'C9': '97.28540713798621',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '980.4114260089906',
                                     'C10': '1009.6451734576028',
                                     'C11': '35.348018297991366',
                                     'C12': '1625.434626662855',
                                     'C2': '899.3108831856057',
                                     'C3': '1183.032552702089',
                                     'C4': '402.0986161964097',
                                     'C5': '81.78864123893297',
                                     'C6': '1115.6819455776936',
                                     'C7': '123.80427864011308',
                                     'C8': '1121.14687497079',
                                     'C9': '0.0024451264081394235',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '1246.782499848912',
                                     'C10': '1987.9907846635351',
                                     'C11': '70.94396914331519',
                                     'C12': '389.15980447937727',
                                     'C2': '2105.7966672265507',
                                     'C3': '1014.3393213554372',
                                     'C4': '1494.6681439414933',
                                     'C5': '362.0173906933171',
                                     'C6': '98.17142420905459',
                                     'C7': '2170.405867933639',
                                     'C8': '97.7318771093979',
                                     'C9': '97.26834016110725',
                                     'Unnamed: 0': 'S6'}},
                         {'source_row': 6,
                          'values': {'C1': '57.1086015288379',
                                     'C10': '524.6154260270081',
                                     'C11': '997.531783848061',
                                     'C12': '104.47794493215576',
                                     'C2': '23.836168859030245',
                                     'C3': '78.10572975165614',
                                     'C4': '742.8068113300407',
                                     'C5': '1926.0796823736941',
                                     'C6': '454.3789956981779',
                                     'C7': '458.2901436941235',
                                     'C8': '465.9307664524444',
                                     'C9': '28.138607069878855',
                                     'Unnamed: 0': 'S7'}},
                         {'source_row': 7,
                          'values': {'C1': '981.2908605082814',
                                     'C10': '275.9784134803488',
                                     'C11': '1228.06989342366',
                                     'C12': '103.48323020673556',
                                     'C2': '120.90130000942015',
                                     'C3': '1625.8206931791087',
                                     'C4': '1267.8229294135008',
                                     'C5': '2569.6446053909003',
                                     'C6': '13.471811837256363',
                                     'C7': '815.1525428026629',
                                     'C8': '253.42349641458793',
                                     'C9': '43.76562945456531',
                                     'Unnamed: 0': 'S8'}},
                         {'source_row': 8,
                          'values': {'C1': '30.532779511898102',
                                     'C10': '1485.9153764236357',
                                     'C11': '29.423790844675874',
                                     'C12': '26.619419605630917',
                                     'C2': '1444.8594995969975',
                                     'C3': '173.5547323639261',
                                     'C4': '1307.3913121142912',
                                     'C5': '965.2012304156898',
                                     'C6': '1843.7769498110006',
                                     'C7': '1483.6408846054035',
                                     'C8': '85.32209952688736',
                                     'C9': '1353.500934450796',
                                     'Unnamed: 0': 'S9'}},
                         {'source_row': 9,
                          'values': {'C1': '94.11093956131819',
                                     'C10': '72.94365832842001',
                                     'C11': '2181.454205937846',
                                     'C12': '973.5515553238279',
                                     'C2': '1422.9971302244805',
                                     'C3': '1470.776907673336',
                                     'C4': '1419.3382251704456',
                                     'C5': '38.94527784177093',
                                     'C6': '72.20112949102915',
                                     'C7': '2040.4605902793303',
                                     'C8': '1542.702557551204',
                                     'C9': '1803.8001691896025',
                                     'Unnamed: 0': 'S10'}},
                         {'source_row': 10,
                          'values': {'C1': '1032.9073835595004',
                                     'C10': '129.79338189093912',
                                     'C11': '1295.09784482215',
                                     'C12': '2330.76820970791',
                                     'C2': '166.30184787458444',
                                     'C3': '1620.4767053028727',
                                     'C4': '64.668341345234',
                                     'C5': '2000.50917314264',
                                     'C6': '0.0028957952749427158',
                                     'C7': '47.038371401381845',
                                     'C8': '52.99221132466169',
                                     'C9': '1115.6336172632205',
                                     'Unnamed: 0': 'S11'}}],
             'returned_rows': 11,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [11, 12],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [11, 12]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    supply_frame = CSVQA_FRAMES['file_1_view_0']
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    I = [row['Unnamed: 0'] for (_, row) in supply_frame.iterrows()]
    J = [row['customer'] for (_, row) in demand_frame.iterrows()]
    d_j = {}
    for (_, row) in demand_frame.iterrows():
        j = row['customer']
        try:
            d_j[j] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {j}: {row['demand']}")
    s_i = {}
    for (_, row) in supply_frame.iterrows():
        i = row['Unnamed: 0']
        try:
            s_i[i] = float(row['supply_capacity'])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for store {i}: {row['supply_capacity']}")
    c_ij = {}
    for (_, row) in cost_frame.iterrows():
        i = row['Unnamed: 0']
        c_ij[i] = {}
        for j in J:
            try:
                c_ij[i][j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid cost value for store {i}, customer {j}: {row[j]}')
    if set(s_i.keys()) != set(I):
        raise ValueError('Supply capacity keys do not match store set I')
    if set(d_j.keys()) != set(J):
        raise ValueError('Demand keys do not match customer set J')
    for i in I:
        if set(c_ij[i].keys()) != set(J):
            raise ValueError(f'Cost matrix row for store {i} does not match customer set J')
    m = gp.Model('Walmart_Transportation')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * quantity_vars[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in I)) >= d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in J)) <= s_i[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)