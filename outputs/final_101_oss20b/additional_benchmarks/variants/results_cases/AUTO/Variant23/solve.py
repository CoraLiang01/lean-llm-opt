CSVQA_DATA = {'query': 'A regional distributor can ship goods from depots to markets. Depot capacities are listed in depot_capacity.csv, market demands are listed in market_demand.csv, per-unit route costs are listed in route_variable_costs.csv, and fixed route activation costs are listed in route_fixed_costs.csv. A route may carry shipments only if it is activated.\n\nFormulate a minimum-cost fixed-charge transportation model. For each depot-market route i-j, define x_ij as the nonnegative shipment quantity and y_ij as a binary variable equal to 1 if route i-j is activated. The objective is to minimize variable shipping cost plus fixed route activation cost. The model should include market demand constraints, depot supply upper-bound constraints, shipment-to-route activation linking constraints using M_ij = min(depot capacity_i, market demand_j), nonnegativity constraints for shipment variables, and binary restrictions for route variables.', 'relationships': [], 'tables': [{'columns': ['Depot', 'SupplyCapacity'], 'file_name': 'depot_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Depot': 'D1', 'SupplyCapacity': '120'}}, {'data_row_number': 1, 'values': {'Depot': 'D2', 'SupplyCapacity': '100'}}, {'data_row_number': 2, 'values': {'Depot': 'D3', 'SupplyCapacity': '140'}}, {'data_row_number': 3, 'values': {'Depot': 'D4', 'SupplyCapacity': '90'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Market', 'Demand'], 'file_name': 'market_demand.csv', 'records': [{'data_row_number': 0, 'values': {'Demand': '65', 'Market': 'M1'}}, {'data_row_number': 1, 'values': {'Demand': '80', 'Market': 'M2'}}, {'data_row_number': 2, 'values': {'Demand': '75', 'Market': 'M3'}}, {'data_row_number': 3, 'values': {'Demand': '95', 'Market': 'M4'}}, {'data_row_number': 4, 'values': {'Demand': '70', 'Market': 'M5'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Depot', 'M1', 'M2', 'M3', 'M4', 'M5'], 'file_name': 'route_variable_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Depot': 'D1', 'M1': '4', 'M2': '5', 'M3': '17', 'M4': '18', 'M5': '16'}}, {'data_row_number': 1, 'values': {'Depot': 'D2', 'M1': '15', 'M2': '14', 'M3': '3', 'M4': '6', 'M5': '17'}}, {'data_row_number': 2, 'values': {'Depot': 'D3', 'M1': '18', 'M2': '16', 'M3': '15', 'M4': '4', 'M5': '5'}}, {'data_row_number': 3, 'values': {'Depot': 'D4', 'M1': '6', 'M2': '7', 'M3': '14', 'M4': '16', 'M5': '13'}}], 'table_id': 'file_2_view_0'}, {'columns': ['Depot', 'M1', 'M2', 'M3', 'M4', 'M5'], 'file_name': 'route_fixed_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Depot': 'D1', 'M1': '240', 'M2': '270', 'M3': '560', 'M4': '590', 'M5': '540'}}, {'data_row_number': 1, 'values': {'Depot': 'D2', 'M1': '520', 'M2': '500', 'M3': '230', 'M4': '280', 'M5': '570'}}, {'data_row_number': 2, 'values': {'Depot': 'D3', 'M1': '610', 'M2': '580', 'M3': '540', 'M4': '260', 'M5': '250'}}, {'data_row_number': 3, 'values': {'Depot': 'D4', 'M1': '290', 'M2': '320', 'M3': '500', 'M4': '550', 'M5': '480'}}], 'table_id': 'file_3_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    depot_table = get_table('file_0_view_0')
    I = [rec['values']['Depot'] for rec in depot_table['records']]
    s_i = {rec['values']['Depot']: float(rec['values']['SupplyCapacity']) for rec in depot_table['records']}
    market_table = get_table('file_1_view_0')
    J = [rec['values']['Market'] for rec in market_table['records']]
    d_j = {rec['values']['Market']: float(rec['values']['Demand']) for rec in market_table['records']}
    var_cost_table = get_table('file_2_view_0')
    c_ij = {}
    for rec in var_cost_table['records']:
        depot = rec['values']['Depot']
        c_ij[depot] = {}
        for market in J:
            if market not in rec['values']:
                raise ValueError(f'Missing cost for depot {depot} market {market}')
            c_ij[depot][market] = float(rec['values'][market])
    fixed_cost_table = get_table('file_3_view_0')
    f_ij = {}
    for rec in fixed_cost_table['records']:
        depot = rec['values']['Depot']
        f_ij[depot] = {}
        for market in J:
            if market not in rec['values']:
                raise ValueError(f'Missing fixed cost for depot {depot} market {market}')
            f_ij[depot][market] = float(rec['values'][market])
    for i in I:
        if i not in s_i:
            raise ValueError(f'Supply capacity missing for depot {i}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Demand missing for market {j}')
    for i in I:
        for j in J:
            if i not in c_ij or j not in c_ij[i]:
                raise ValueError(f'Variable cost missing for depot {i} market {j}')
            if i not in f_ij or j not in f_ij[i]:
                raise ValueError(f'Fixed cost missing for depot {i} market {j}')
    M_ij = {}
    for i in I:
        M_ij[i] = {}
        for j in J:
            M_ij[i][j] = min(s_i[i], d_j[j])
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = model.addVars([(i, j) for i in I for j in J], vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((c_ij[i][j] * x[i, j] + f_ij[i][j] * y[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        model.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == d_j[j], name=f'demand_{j}')
    for i in I:
        model.addConstr(gurobipy.quicksum((x[i, j] for j in J)) <= s_i[i], name=f'supply_{i}')
    for i in I:
        for j in J:
            model.addConstr(x[i, j] <= M_ij[i][j] * y[i, j], name=f'link_{i}_{j}')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {model.ObjVal:.4f}')
        for var in model.getVars():
            print(f'{var.VarName} = {var.X:.4f}')
    return model
m = solve_problem()
