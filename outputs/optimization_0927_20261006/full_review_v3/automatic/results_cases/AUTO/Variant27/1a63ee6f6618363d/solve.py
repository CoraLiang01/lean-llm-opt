import gurobipy as gp
from gurobipy import GRB
workers = ['W00', 'W18', 'W02', 'W17', 'W10', 'W05', 'W11', 'W14', 'W01', 'W24', 'W16']
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09']
skill_map = {'Junior': 1, 'Intermediate': 2, 'Senior': 3, 'Expert': 4}
worker_skill = {'W00': 'Intermediate', 'W18': 'Intermediate', 'W02': 'Junior', 'W17': 'Junior', 'W10': 'Expert', 'W05': 'Senior', 'W11': 'Expert', 'W14': 'Expert', 'W01': 'Intermediate', 'W24': 'Expert', 'W16': 'Senior'}
project_required_skill = {'P00': 'Junior', 'P01': 'Junior', 'P02': 'Junior', 'P03': 'Intermediate', 'P04': 'Junior', 'P05': 'Junior', 'P06': 'Junior', 'P07': 'Expert', 'P08': 'Junior', 'P09': 'Intermediate'}
cost = {'W00': {'P01': 1116, 'P02': 414, 'P04': 554, 'P05': 113, 'P07': 11, 'P08': 853, 'P09': 1142}, 'W18': {'P00': 238, 'P02': 1085, 'P03': 529, 'P04': 582, 'P05': 889, 'P06': 139, 'P07': 18, 'P08': 630, 'P09': 538}, 'W02': {'P00': 330, 'P01': 193, 'P03': 4, 'P04': 431, 'P06': 1130, 'P07': 15, 'P08': 644, 'P09': 9}, 'W17': {'P00': 538, 'P02': 1359, 'P03': 20, 'P04': 1366, 'P05': 702, 'P06': 122, 'P07': 13, 'P08': 358}, 'W10': {'P00': 1079, 'P01': 1128, 'P02': 758, 'P05': 108, 'P07': 423, 'P08': 744, 'P09': 347}, 'W05': {'P00': 325, 'P01': 772, 'P02': 1042, 'P04': 1394, 'P06': 374, 'P07': 2, 'P08': 1140, 'P09': 127}, 'W11': {'P00': 1143, 'P01': 841, 'P02': 920, 'P03': 122, 'P04': 634, 'P06': 1250, 'P07': 836, 'P08': 180, 'P09': 1105}, 'W14': {'P01': 687, 'P02': 126, 'P03': 547, 'P04': 125, 'P05': 1358, 'P06': 1391, 'P07': 814, 'P08': 1362, 'P09': 962}, 'W01': {'P01': 170, 'P03': 919, 'P04': 660, 'P05': 1314, 'P06': 668, 'P07': 4, 'P08': 462, 'P09': 614}, 'W24': {'P00': 784, 'P01': 1178, 'P02': 1171, 'P04': 103, 'P05': 137, 'P06': 832, 'P07': 1279, 'P08': 893, 'P09': 351}, 'W16': {'P00': 265, 'P01': 1114, 'P03': 1209, 'P04': 231, 'P05': 221, 'P07': 17, 'P08': 135}}
eligible_pairs = []
for w in workers:
    w_skill = skill_map[worker_skill[w]]
    for p in projects:
        p_skill = skill_map[project_required_skill[p]]
        if w_skill >= p_skill and p in cost.get(w, {}):
            eligible_pairs.append((w, p))
for p in projects:
    if not any(((w, p) in eligible_pairs for w in workers)):
        raise ValueError(f'No eligible worker for project {p}')
for (w, p) in eligible_pairs:
    if p not in cost[w]:
        raise ValueError(f'Missing cost for allowed assignment ({w},{p})')
m = gp.Model('assignment')
x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in eligible_pairs)), GRB.MINIMIZE)
for p in projects:
    m.addConstr(gp.quicksum((x_vars[w, p] for w in workers if (w, p) in eligible_pairs)) == 1, name='')
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, p] for p in projects if (w, p) in eligible_pairs)) <= 1, name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')