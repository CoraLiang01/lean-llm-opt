LEGACY_OBSERVATION = '{"values": {"Unnamed: 0": "Depot", "Depot": "0", "A": "28", "B": "41", "C": "63", "D": "39", "E": "38", "F": "45", "G": "35", "H": "28", "I": "44", "J": "35"}}\n{"values": {"Unnamed: 0": "A", "Depot": "28", "A": "0", "B": "27", "C": "87", "D": "35", "E": "65", "F": "63", "G": "41", "H": "39", "I": "43", "J": "20"}}\n{"values": {"Unnamed: 0": "B", "Depot": "41", "A": "27", "B": "0", "C": "81", "D": "13", "E": "77", "F": "54", "G": "25", "H": "63", "I": "70", "J": "7"}}\n{"values": {"Unnamed: 0": "C", "Depot": "63", "A": "87", "B": "81", "C": "0", "D": "69", "E": "53", "F": "28", "G": "57", "H": "83", "I": "102", "J": "81"}}\n{"values": {"Unnamed: 0": "D", "Depot": "39", "A": "35", "B": "13", "C": "69", "D": "0", "E": "72", "F": "41", "G": "12", "H": "64", "I": "75", "J": "17"}}\n{"values": {"Unnamed: 0": "E", "Depot": "38", "A": "65", "B": "77", "C": "53", "D": "72", "E": "0", "F": "53", "G": "64", "H": "39", "I": "58", "J": "72"}}\n{"values": {"Unnamed: 0": "F", "Depot": "45", "A": "63", "B": "54", "C": "28", "D": "41", "E": "53", "F": "0", "G": "29", "H": "70", "I": "88", "J": "54"}}\n{"values": {"Unnamed: 0": "G", "Depot": "35", "A": "41", "B": "25", "C": "57", "D": "12", "E": "64", "F": "29", "G": "0", "H": "63", "I": "76", "J": "27"}}\n{"values": {"Unnamed: 0": "H", "Depot": "28", "A": "39", "B": "63", "C": "83", "D": "64", "E": "39", "F": "70", "G": "63", "H": "0", "I": "20", "J": "56"}}\n{"values": {"Unnamed: 0": "I", "Depot": "44", "A": "43", "B": "70", "C": "102", "D": "75", "E": "58", "F": "88", "G": "76", "H": "20", "I": "0", "J": "63"}}\n{"values": {"Unnamed: 0": "J", "Depot": "35", "A": "20", "B": "7", "C": "81", "D": "17", "E": "72", "F": "54", "G": "27", "H": "56", "I": "63", "J": "0"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Unnamed: 0': 'Depot', 'Depot': '0', 'A': '28', 'B': '41', 'C': '63', 'D': '39', 'E': '38', 'F': '45', 'G': '35', 'H': '28', 'I': '44', 'J': '35'}}, {'source': '', 'values': {'Unnamed: 0': 'A', 'Depot': '28', 'A': '0', 'B': '27', 'C': '87', 'D': '35', 'E': '65', 'F': '63', 'G': '41', 'H': '39', 'I': '43', 'J': '20'}}, {'source': '', 'values': {'Unnamed: 0': 'B', 'Depot': '41', 'A': '27', 'B': '0', 'C': '81', 'D': '13', 'E': '77', 'F': '54', 'G': '25', 'H': '63', 'I': '70', 'J': '7'}}, {'source': '', 'values': {'Unnamed: 0': 'C', 'Depot': '63', 'A': '87', 'B': '81', 'C': '0', 'D': '69', 'E': '53', 'F': '28', 'G': '57', 'H': '83', 'I': '102', 'J': '81'}}, {'source': '', 'values': {'Unnamed: 0': 'D', 'Depot': '39', 'A': '35', 'B': '13', 'C': '69', 'D': '0', 'E': '72', 'F': '41', 'G': '12', 'H': '64', 'I': '75', 'J': '17'}}, {'source': '', 'values': {'Unnamed: 0': 'E', 'Depot': '38', 'A': '65', 'B': '77', 'C': '53', 'D': '72', 'E': '0', 'F': '53', 'G': '64', 'H': '39', 'I': '58', 'J': '72'}}, {'source': '', 'values': {'Unnamed: 0': 'F', 'Depot': '45', 'A': '63', 'B': '54', 'C': '28', 'D': '41', 'E': '53', 'F': '0', 'G': '29', 'H': '70', 'I': '88', 'J': '54'}}, {'source': '', 'values': {'Unnamed: 0': 'G', 'Depot': '35', 'A': '41', 'B': '25', 'C': '57', 'D': '12', 'E': '64', 'F': '29', 'G': '0', 'H': '63', 'I': '76', 'J': '27'}}, {'source': '', 'values': {'Unnamed: 0': 'H', 'Depot': '28', 'A': '39', 'B': '63', 'C': '83', 'D': '64', 'E': '39', 'F': '70', 'G': '63', 'H': '0', 'I': '20', 'J': '56'}}, {'source': '', 'values': {'Unnamed: 0': 'I', 'Depot': '44', 'A': '43', 'B': '70', 'C': '102', 'D': '75', 'E': '58', 'F': '88', 'G': '76', 'H': '20', 'I': '0', 'J': '63'}}, {'source': '', 'values': {'Unnamed: 0': 'J', 'Depot': '35', 'A': '20', 'B': '7', 'C': '81', 'D': '17', 'E': '72', 'F': '54', 'G': '27', 'H': '56', 'I': '63', 'J': '0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
nodes = ['Depot', 'A', 'B', 'C']
d = {}
for rec in records:
    row = rec['values']
    i = row['Unnamed: 0']
    if i in nodes:
        d[i] = {}
        for j in nodes:
            val = row[j]
            d[i][j] = float(val)
for i in nodes:
    for j in nodes:
        if i != j and (i not in d or j not in d[i]):
            raise ValueError(f'Missing distance from {i} to {j}')
customers = ['A', 'B', 'C']
m = gp.Model('TSP_Courier')
x = m.addVars(nodes, nodes, vtype=GRB.BINARY, lb=0, ub=1, obj=0, name='')
for i in nodes:
    x[i, i].ub = 0
u = m.addVars(customers, lb=1, ub=3, vtype=GRB.CONTINUOUS, name='')
u_depot = m.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS, name='u_depot')
m.setObjective(gp.quicksum((d[i][j] * x[i, j] for i in nodes for j in nodes if i != j)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in nodes if j != i)) == 1 for i in nodes), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in nodes if i != j)) == 1 for j in nodes), name='')
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u[i] - u[j] + 3 * x[i, j] <= 2, name=f'mtz_{i}_{j}')
m.addConstr(u_depot == 0, name='u_depot_fix')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')