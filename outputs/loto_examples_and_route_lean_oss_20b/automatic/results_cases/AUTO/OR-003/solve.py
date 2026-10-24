CSVQA_DATA = {'query': 'There are multiple suppliers responsible for delivering essential goods daily to customer groups located in different regions. Each supplier has a specific daily supply capacity, detailed in "supply_capacity.csv", while the daily demand of each customer group is recorded in "customer_demand.csv". The transportation cost per unit of goods from each supplier to each customer group is provided in "transportation_costs.csv". The objective is to determine the optimal transportation plan, specifying how goods should be allocated from each supplier to each customer group, ensuring that all demands are met without exceeding the supply capacity of any supplier, while minimizing the total transportation cost.', 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'}, 'row_id_column': 'Unnamed: 0', 'type': 'matrix'}], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '216'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '168'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '264'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '216'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '216'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '192'}}, {'data_row_number': 6, 'values': {'customer': 'C7', 'demand': '144'}}, {'data_row_number': 7, 'values': {'customer': 'C8', 'demand': '168'}}, {'data_row_number': 8, 'values': {'customer': 'C9', 'demand': '168'}}, {'data_row_number': 9, 'values': {'customer': 'C10', 'demand': '168'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'supply_capacity': '288'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'supply_capacity': '288'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'supply_capacity': '264'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'supply_capacity': '264'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'supply_capacity': '216'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'supply_capacity': '216'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'S7', 'supply_capacity': '168'}}, {'data_row_number': 7, 'values': {'Unnamed: 0': 'S8', 'supply_capacity': '216'}}, {'data_row_number': 8, 'values': {'Unnamed: 0': 'S9', 'supply_capacity': '240'}}, {'data_row_number': 9, 'values': {'Unnamed: 0': 'S10', 'supply_capacity': '168'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '590.3648136504455', 'C10': '58.895473920714416', 'C2': '23.669172607322494', 'C3': '88.89005869765714', 'C4': '497.52228807074613', 'C5': '466.09034321595647', 'C6': '29.022096827063212', 'C7': '23.675244833973835', 'C8': '23.677760288117437', 'C9': '0.3118394914937161', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '2042.0715001593626', 'C10': '67.29170751036096', 'C2': '2133.978484314172', 'C3': '705.15912033561', 'C4': '101.59454516295598', 'C5': '2052.937657376311', 'C6': '1738.754951414345', 'C7': '101.61094965654742', 'C8': '101.61062174376057', 'C9': '122.45214268700826', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '22.297222160217984', 'C10': '1008.6717394620979', 'C2': '497.9271939314995', 'C3': '1653.0828862073263', 'C4': '23.68545123339267', 'C5': '1386.0807887282344', 'C6': '26.13715280763276', 'C7': '497.6220482906461', 'C8': '498.0935847133144', 'C9': '865.3816296318804', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '960.7814533858373', 'C10': '1351.8189071012546', 'C2': '49.128300053752405', 'C3': '1324.238697073691', 'C4': '1032.2095478151716', 'C5': '0.07804725392720868', 'C6': '53.308268726049285', 'C7': '49.1364167175093', 'C8': '1031.8214424894484', 'C9': '466.00495307991264', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '1471.2721666392908', 'C10': '1094.6695960752636', 'C2': '85.6956072820555', 'C3': '38.89266823851542', 'C4': '1542.0500358120464', 'C5': '112.20514372003504', 'C6': '82.3702016356405', 'C7': '1542.3399196620971', 'C8': '85.69238745806277', 'C9': '1924.9360769614245', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '191.9058726130392', 'C10': '929.807168280051', 'C2': '158.50401031820448', 'C3': '91.02045349777458', 'C4': '184.44747201726193', 'C5': '968.146798696633', 'C6': '284.1076062070199', 'C7': '8.791061587686942', 'C8': '158.70523835548545', 'C9': '27.943874345249665', 'Unnamed: 0': 'S6'}}, {'data_row_number': 6, 'values': {'C1': '81.23891457326876', 'C10': '849.9799406578097', 'C2': '0.3744642223062507', 'C3': '2079.46686537067', 'C4': '0.3065671755503025', 'C5': '1031.7772962191823', 'C6': '7.203964492497209', 'C7': '0.07623072241762692', 'C8': '0.032473879548006554', 'C9': '23.685827966421357', 'Unnamed: 0': 'S7'}}, {'data_row_number': 7, 'values': {'C1': '56.099310965461356', 'C10': '1348.8366145919845', 'C2': '935.6143108671334', 'C3': '73.08824617002863', 'C4': '52.00392409272077', 'C5': '4.025792388934198', 'C6': '1002.2327657984296', 'C7': '935.7766029588662', 'C8': '935.7007252277288', 'C9': '612.8698719325438', 'Unnamed: 0': 'S8'}}, {'data_row_number': 8, 'values': {'C1': '4.502283326860296', 'C10': '40.46575554562011', 'C2': '0.3899585342810754', 'C3': '1782.4662178163346', 'C4': '0.006345906612718274', 'C5': '1031.9910114913148', 'C6': '129.50665619510303', 'C7': '0.2118319573481142', 'C8': '0.645730107353115', 'C9': '497.62723911435927', 'Unnamed: 0': 'S9'}}, {'data_row_number': 9, 'values': {'C1': '333.6869270439132', 'C10': '941.7526365563969', 'C2': '277.4719386113677', 'C3': '86.02096892455509', 'C4': '277.30836609256806', 'C5': '1004.4649084520337', 'C6': '19.950336815857597', 'C7': '13.202073690286834', 'C8': '238.14321521805866', 'C9': '411.0580332361589', 'Unnamed: 0': 'S10'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    sup_table = get_table('file_1_view_0')
    S = []
    supply_capacity = {}
    for rec in sup_table['records']:
        s_id = rec['values']['Unnamed: 0']
        S.append(s_id)
        try:
            supply_capacity[s_id] = int(rec['values']['supply_capacity'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid supply_capacity for supplier {s_id}')
    cust_table = get_table('file_0_view_0')
    C = []
    demand = {}
    for rec in cust_table['records']:
        c_id = rec['values']['customer']
        C.append(c_id)
        try:
            demand[c_id] = int(rec['values']['demand'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid demand for customer {c_id}')
    cost_table = get_table('file_2_view_0')
    cost = {}
    for rec in cost_table['records']:
        s_id = rec['values']['Unnamed: 0']
        if s_id not in S:
            raise ValueError(f'Supplier {s_id} in cost matrix not in supplier list')
        cost[s_id] = {}
        for c_id in C:
            try:
                cost[s_id][c_id] = float(rec['values'][c_id])
            except (KeyError, ValueError):
                raise ValueError(f'Missing or invalid cost for supplier {s_id} to customer {c_id}')
    for s_id in S:
        if s_id not in cost:
            raise ValueError(f'Supplier {s_id} missing in cost matrix')
        for c_id in C:
            if c_id not in cost[s_id]:
                raise ValueError(f'Cost missing for supplier {s_id} to customer {c_id}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(s, c) for s in S for c in C], vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((cost[s][c] * x[s, c] for s in S for c in C)), GRB.MINIMIZE)
    for s in S:
        model.addConstr(gurobipy.quicksum((x[s, c] for c in C)) <= supply_capacity[s], name=f'supply_{s}')
    for c in C:
        model.addConstr(gurobipy.quicksum((x[s, c] for s in S)) >= demand[c], name=f'demand_{c}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()
