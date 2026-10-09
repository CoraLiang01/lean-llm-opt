import gurobipy as gp
from gurobipy import GRB
workers = ['W00', 'W18', 'W02', 'W17', 'W10', 'W05', 'W11', 'W14', 'W01', 'W24', 'W16']
projects = ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09']
worker_skill = {'W00': 'Intermediate', 'W18': 'Intermediate', 'W02': 'Junior', 'W17': 'Junior', 'W10': 'Expert', 'W05': 'Senior', 'W11': 'Expert', 'W14': 'Expert', 'W01': 'Intermediate', 'W24': 'Expert', 'W16': 'Senior'}
project_required_skill = {'P00': 'Junior', 'P01': 'Junior', 'P02': 'Junior', 'P03': 'Intermediate', 'P04': 'Junior', 'P05': 'Junior', 'P06': 'Junior', 'P07': 'Expert', 'P08': 'Junior', 'P09': 'Intermediate'}
skill_level = {'Junior': 0, 'Intermediate': 1, 'Senior': 2, 'Expert': 3}
cost = {'W00': {'P01': 1116, 'P02': 414, 'P04': 554, 'P05': 113, 'P08': 853, 'P09': 1142}, 'W18': {'P00': 238, 'P02': 1085, 'P03': 529, 'P04': 582, 'P05': 889, 'P06': 139, 'P08': 630, 'P09': 538}, 'W02': {'P00': 330, 'P01': 193, 'P04': 431, 'P06': 1130, 'P08': 644}, 'W17': {'P00': 538, 'P02': 1359, 'P04': 1366, 'P05': 702, 'P06': 122, 'P08': 358}, 'W10': {'P00': 1079, 'P01': 1128, 'P02': 758, 'P05': 108, 'P07': 423, 'P08': 744, 'P09': 347}, 'W05': {'P00': 325, 'P01': 772, 'P02': 1042, 'P04': 1394, 'P06': 374, 'P08': 1140, 'P09': 127}, 'W11': {'P00': 1143, 'P01': 841, 'P02': 920, 'P03': 122, 'P04': 634, 'P06': 1250, 'P07': 836, 'P08': 180, 'P09': 1105}, 'W14': {'P01': 687, 'P02': 126, 'P03': 547, 'P04': 125, 'P05': 1358, 'P06': 1391, 'P07': 814, 'P08': 1362, 'P09': 962}, 'W01': {'P01': 170, 'P03': 919, 'P04': 660, 'P05': 1314, 'P06': 668, 'P08': 462, 'P09': 614}, 'W24': {'P00': 784, 'P01': 1178, 'P02': 1171, 'P04': 103, 'P05': 137, 'P06': 832, 'P07': 1279, 'P08': 893, 'P09': 351}, 'W16': {'P00': 265, 'P01': 1114, 'P03': 1209, 'P04': 231, 'P05': 221, 'P08': 135}}
eligible_pairs = []
for w in workers:
    for p in projects:
        if p in cost.get(w, {}):
            if skill_level[worker_skill[w]] >= skill_level[project_required_skill[p]]:
                eligible_pairs.append((w, p))
for w in cost:
    for p in cost[w]:
        if (w, p) not in eligible_pairs:
            raise ValueError(f'Cost entry ({w},{p}) is not eligible by skill or project definition.')
for p in projects:
    if not any(((w, p) in eligible_pairs for w in workers)):
        raise ValueError(f'Project {p} has no eligible workers.')
m = gp.Model('assignment')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(eligible_pairs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w][p] * x_vars[w, p] for (w, p) in eligible_pairs)), GRB.MINIMIZE)
for p in projects:
    m.addConstr(gp.quicksum((x_vars[w, p] for w in workers if (w, p) in eligible_pairs)) == 1, name='prj_' + p)
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, p] for p in projects if (w, p) in eligible_pairs)) <= 1, name='wrk_' + w)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')