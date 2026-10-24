CSVQA_DATA = {'query': 'Educational planners must be able to address the travel-time issue, while striving to satisfy other requirements. One approach to the problem is to consider racial balance, school capacity, and population of the school district as constraints, while striving to transport the children with as little inconvenience to them as possible. Suppose the Capital City School District has two elementary schools, with individual capacities in school_capacity.csv, and 31 neighborhoods with elementary school populations summarized in neighborhoods_population.csv. The distance in miles from each school to each neighborhood is given in distance.csv. The district has decided to integrate by school rather than by grade level. A school will be considered to be in racial balance when its white-student percentage deviates by no more than 10 percentage points from the district ratio of 60 percent white and 40 percent nonwhite. The district wishes to devise a busing plan that assigns white and nonwhite pupils from each neighborhood to schools, respects school capacities and neighborhood populations, and minimizes the total distance the youngsters must travel.', 'relationships': [{'column_axis': {'id_column': 'Neighborhood', 'table_id': 'file_1_view_0'}, 'matrix_table_id': 'file_2_view_0', 'row_axis': {'id_column': 'School', 'table_id': 'file_0_view_0'}, 'row_id_column': 'School', 'type': 'matrix'}], 'tables': [{'columns': ['School', 'Capacity'], 'file_name': 'school_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Capacity': '2028', 'School': 'I'}}, {'data_row_number': 1, 'values': {'Capacity': '1560', 'School': 'II'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Neighborhood', 'Population_White', 'Population_NonWhite'], 'file_name': 'neighborhoods_population.csv', 'records': [{'data_row_number': 0, 'values': {'Neighborhood': 'N01', 'Population_NonWhite': '22', 'Population_White': '78'}}, {'data_row_number': 1, 'values': {'Neighborhood': 'N02', 'Population_NonWhite': '33', 'Population_White': '57'}}, {'data_row_number': 2, 'values': {'Neighborhood': 'N03', 'Population_NonWhite': '63', 'Population_White': '47'}}, {'data_row_number': 3, 'values': {'Neighborhood': 'N04', 'Population_NonWhite': '22', 'Population_White': '78'}}, {'data_row_number': 4, 'values': {'Neighborhood': 'N05', 'Population_NonWhite': '33', 'Population_White': '57'}}, {'data_row_number': 5, 'values': {'Neighborhood': 'N06', 'Population_NonWhite': '63', 'Population_White': '47'}}, {'data_row_number': 6, 'values': {'Neighborhood': 'N07', 'Population_NonWhite': '22', 'Population_White': '78'}}, {'data_row_number': 7, 'values': {'Neighborhood': 'N08', 'Population_NonWhite': '33', 'Population_White': '57'}}, {'data_row_number': 8, 'values': {'Neighborhood': 'N09', 'Population_NonWhite': '64', 'Population_White': '46'}}, {'data_row_number': 9, 'values': {'Neighborhood': 'N10', 'Population_NonWhite': '23', 'Population_White': '77'}}, {'data_row_number': 10, 'values': {'Neighborhood': 'N11', 'Population_NonWhite': '34', 'Population_White': '56'}}, {'data_row_number': 11, 'values': {'Neighborhood': 'N12', 'Population_NonWhite': '64', 'Population_White': '46'}}, {'data_row_number': 12, 'values': {'Neighborhood': 'N13', 'Population_NonWhite': '23', 'Population_White': '77'}}, {'data_row_number': 13, 'values': {'Neighborhood': 'N14', 'Population_NonWhite': '34', 'Population_White': '56'}}, {'data_row_number': 14, 'values': {'Neighborhood': 'N15', 'Population_NonWhite': '64', 'Population_White': '46'}}, {'data_row_number': 15, 'values': {'Neighborhood': 'N16', 'Population_NonWhite': '23', 'Population_White': '77'}}, {'data_row_number': 16, 'values': {'Neighborhood': 'N17', 'Population_NonWhite': '34', 'Population_White': '56'}}, {'data_row_number': 17, 'values': {'Neighborhood': 'N18', 'Population_NonWhite': '64', 'Population_White': '46'}}, {'data_row_number': 18, 'values': {'Neighborhood': 'N19', 'Population_NonWhite': '23', 'Population_White': '77'}}, {'data_row_number': 19, 'values': {'Neighborhood': 'N20', 'Population_NonWhite': '34', 'Population_White': '56'}}, {'data_row_number': 20, 'values': {'Neighborhood': 'N21', 'Population_NonWhite': '64', 'Population_White': '46'}}, {'data_row_number': 21, 'values': {'Neighborhood': 'N22', 'Population_NonWhite': '23', 'Population_White': '77'}}, {'data_row_number': 22, 'values': {'Neighborhood': 'N23', 'Population_NonWhite': '34', 'Population_White': '56'}}, {'data_row_number': 23, 'values': {'Neighborhood': 'N24', 'Population_NonWhite': '64', 'Population_White': '46'}}, {'data_row_number': 24, 'values': {'Neighborhood': 'N25', 'Population_NonWhite': '23', 'Population_White': '77'}}, {'data_row_number': 25, 'values': {'Neighborhood': 'N26', 'Population_NonWhite': '34', 'Population_White': '56'}}, {'data_row_number': 26, 'values': {'Neighborhood': 'N27', 'Population_NonWhite': '64', 'Population_White': '46'}}, {'data_row_number': 27, 'values': {'Neighborhood': 'N28', 'Population_NonWhite': '23', 'Population_White': '77'}}, {'data_row_number': 28, 'values': {'Neighborhood': 'N29', 'Population_NonWhite': '34', 'Population_White': '56'}}, {'data_row_number': 29, 'values': {'Neighborhood': 'N30', 'Population_NonWhite': '64', 'Population_White': '46'}}, {'data_row_number': 30, 'values': {'Neighborhood': 'N31', 'Population_NonWhite': '46', 'Population_White': '74'}}], 'table_id': 'file_1_view_0'}, {'columns': ['School', 'N01', 'N02', 'N03', 'N04', 'N05', 'N06', 'N07', 'N08', 'N09', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'N16', 'N17', 'N18', 'N19', 'N20', 'N21', 'N22', 'N23', 'N24', 'N25', 'N26', 'N27', 'N28', 'N29', 'N30', 'N31'], 'file_name': 'distance.csv', 'records': [{'data_row_number': 0, 'values': {'N01': '1.25', 'N02': '1.3', 'N03': '1.35', 'N04': '1.4', 'N05': '1.45', 'N06': '1.5', 'N07': '1.55', 'N08': '1.6', 'N09': '1.65', 'N10': '1.7', 'N11': '1.75', 'N12': '1.8', 'N13': '1.85', 'N14': '1.9', 'N15': '1.95', 'N16': '2.0', 'N17': '3.08', 'N18': '3.16', 'N19': '3.24', 'N20': '3.32', 'N21': '3.4', 'N22': '3.48', 'N23': '3.56', 'N24': '3.64', 'N25': '3.7199999999999998', 'N26': '3.8', 'N27': '3.88', 'N28': '3.96', 'N29': '4.04', 'N30': '4.12', 'N31': '4.2', 'School': 'I'}}, {'data_row_number': 1, 'values': {'N01': '3.08', 'N02': '3.16', 'N03': '3.24', 'N04': '3.32', 'N05': '3.4', 'N06': '3.48', 'N07': '3.56', 'N08': '3.64', 'N09': '3.7199999999999998', 'N10': '3.8', 'N11': '3.88', 'N12': '3.96', 'N13': '4.04', 'N14': '4.12', 'N15': '4.2', 'N16': '4.28', 'N17': '1.25', 'N18': '1.3', 'N19': '1.35', 'N20': '1.4', 'N21': '1.45', 'N22': '1.5', 'N23': '1.55', 'N24': '1.6', 'N25': '1.65', 'N26': '1.7', 'N27': '1.75', 'N28': '1.8', 'N29': '1.85', 'N30': '1.9', 'N31': '1.95', 'School': 'II'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    cap_table = get_table('file_0_view_0')
    Cap_s = {}
    for rec in cap_table['records']:
        school = rec['values']['School']
        Cap_s[school] = int(rec['values']['Capacity'])
    pop_table = get_table('file_1_view_0')
    Pop_n_W = {}
    Pop_n_NW = {}
    for rec in pop_table['records']:
        n = rec['values']['Neighborhood']
        Pop_n_W[n] = int(rec['values']['Population_White'])
        Pop_n_NW[n] = int(rec['values']['Population_NonWhite'])
    dist_table = get_table('file_2_view_0')
    Dist_s_n = {}
    for rec in dist_table['records']:
        school = rec['values']['School']
        for n in Pop_n_W.keys():
            Dist_s_n[school, n] = float(rec['values'][n])
    S = list(Cap_s.keys())
    N = list(Pop_n_W.keys())
    R = ['W', 'NW']
    for s in S:
        if s not in Cap_s:
            raise ValueError(f'Missing capacity for school {s}')
    for n in N:
        if n not in Pop_n_W or n not in Pop_n_NW:
            raise ValueError(f'Missing population for neighborhood {n}')
    for s in S:
        for n in N:
            if (s, n) not in Dist_s_n:
                raise ValueError(f'Missing distance for school {s} and neighborhood {n}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(s, n, r) for s in S for n in N for r in R], vtype=GRB.INTEGER, lb=0, name='')
    obj = gurobipy.quicksum((Dist_s_n[s, n] * (x[s, n, 'W'] + x[s, n, 'NW']) for s in S for n in N))
    m.setObjective(obj, GRB.MINIMIZE)
    for s in S:
        m.addConstr(gurobipy.quicksum((x[s, n, r] for n in N for r in R)) <= Cap_s[s], name=f'cap_{s}')
    for n in N:
        m.addConstr(gurobipy.quicksum((x[s, n, r] for s in S for r in R)) == Pop_n_W[n] + Pop_n_NW[n], name=f'pop_{n}')
    for s in S:
        total = gurobipy.quicksum((x[s, n, r] for n in N for r in R))
        white = gurobipy.quicksum((x[s, n, 'W'] for n in N))
        m.addConstr(0.5 * total <= white, name=f'race_low_{s}')
        m.addConstr(white <= 0.7 * total, name=f'race_high_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()
