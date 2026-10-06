"""Vary representation and wording without changing the optimization problems."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'benchmark_dataset'
COPY=ROOT/'Test_Dataset/Large-scale-or/Large-scale-or-101_冗余列35个 copy.csv'
spec=importlib.util.spec_from_file_location('validator',BASE/'reference/validate.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
core=v.core

def write(p, rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,quoting=csv.QUOTE_ALL);w.writeheader();w.writerows(rows)

def path_for(out,i):return out/f'batch_{i%6+1:02d}'/f'export_{i+1:02d}.csv'

def economic_signature(t):
    fx={r['currency']:(int(r['usd_cents_numerator']),int(r['denominator'])) for r in t['fx']}
    # FX is a representation detail; compare the resulting economic coefficients.
    keep={k:sorted(json.dumps(r,sort_keys=True) for r in t[k]) for k in ['item','item_fee','usage','capacity_ledger','category','incompatible','requires','bundle']}
    keep['benefit']=dict(core.money(t['benefit'],fx,'item_ref'))
    return keep

OPENERS={
 'RA12':'Prepare the produce order for the CENTRAL_FRESH supermarket. Choose an integer number of cases of each product to maximize the net return from this order.',
 'RA1':'The bakery needs an order for its next production run. How many units of each bread option should it order to earn the highest net benefit within the available storage, staff time and energy?',
 'RA2':'A vehicle dealer is reconciling its inventory records before placing an order. For business unit NORTH, use the records effective on or before 2026-03-12 and find the replenishment quantities with the highest net return.',
 'RA3':'RIVERSIDE_AUTO has received vehicle offers. Select quantities of the offered configurations for its next delivery, maximizing the total contribution after fixed preparation charges.',
 'RA4':'The Oslo dealership is reviewing a multi-currency replenishment proposal. Recover the valid OSLO_NEW_CARS records as of 2026-05-07, then find the vehicle order with the largest net benefit.',
 'RA7':'Choose a portfolio of standardized development modules in New York. Each item is an indivisible module option. Maximize the portfolio return after setup charges, subject to construction-material storage, labor and energy availability.',
 'RA10':'Stock the three display sections of MARKET_SQUARE. Each item_ref is a product-and-section option identified by location_id. Determine the number of packs to display in each option, maximizing net merchandising benefit. Each section has its own capacity; unused space cannot be transferred to another section.',
 'RA13':'Allocate licensed game-edition packages to PC, CONSOLE and MOBILE. Each item_ref is one edition-and-platform option, with its platform in location_id. Maximize the licensing return after fees. Memory capacity is separate for each platform and cannot be pooled.',
 'RA14':'Plan air-conditioner placement at FC_EAST_HVAC. Each item_ref represents a model assigned to the storage area in location_id. Choose the integer quantities that maximize net value. Each area has its own volume limit and cannot borrow capacity from another area.'}

def query(name,mode):
    s=OPENERS[name]+' '
    if mode=='events':
        scope='tenant' if name=='RA2' else 'dealership_id'
        s+=(f'For each ({scope}, table, record_id), exclude future events, keep the highest integer revision and discard the record if that revision is DELETE. '
            'Identical retransmissions count once. This selection rule applies to every table before joining or summing records. ')
    else:s+='The supplied tables describe the current plan; use their rows directly. '
    if name in ['RA1','RA3','RA12','RA14']:
        s+='The item rows give unit_benefit_cents and item_fee_cents: earn the former per unit and pay the latter once for any positive quantity. '
    elif name in ['RA4','RA13']:
        s+='For each item, sum the signed benefit components after converting amount * usd_cents_numerator / denominator using fx. This is the benefit per unit in USD cents. Deduct item_fee once for each option with positive quantity. '
    else:s+='The benefit rows contain signed amount_cents components; sum them for each item to obtain its per-unit benefit. Deduct item_fee once for each option with positive quantity. '
    if name in ['RA3','RA7','RA14']:
        s+='The usage amounts and capacity_ledger amounts are already in matching base units for each resource. '
    elif name=='RA13':s+='Convert GB to MB using 1000 MB per GB. '
    elif name=='RA10':s+='Convert liters to ml using 1000 ml per liter. '
    else:s+='Use 1000 ml per liter, 60 minutes per hour and 1000 wh per kwh when comparing resource use with capacity. '
    if name in ['RA1','RA12']:
        s+=('Only authorized=1 options can be ordered. Quantities are nonnegative integers: zero, or minimum_lot through maximum_order. '
            'For each resource, total per-unit usage times quantities must fit the signed sum of capacity_ledger entries. '
            'Meet every category minimum_quantity and maximum_quantity, and pay its activation_fee_cents once if any option is ordered. '
            'Do not select both members of an incompatible pair. An ordered item must have its requires prerequisite ordered as well; no quantity ratio applies. '
            'A bundle earns bonus_cents once only when both options have positive quantities; an unauthorized option cannot trigger a bonus. ')
    elif name in ['RA3','RA4','RA2']:
        s+=('An authorized option may be rejected or accepted in an integer lot between minimum_lot and maximum_order; unauthorized options must have zero quantity. '
            'The signed capacity_ledger total is the limit for each resource, and usage is per ordered unit. '
            'Category quantity limits are unconditional. The category activation fee is charged once when that category is used. '
            'Respect all incompatible pairs and requires dependencies: the latter require a positive prerequisite quantity, not a quantity ratio. '
            'Award each bundle bonus once when both options are ordered, and zero otherwise, including when either option is unauthorized. ')
    else:
        s+=('Choose zero for unauthorized options. For any other option, choose zero or an integer from minimum_lot to maximum_order. '
            'Sum the signed capacity_ledger entries separately by resource; the total usage of that resource must stay within this amount. '
            'Across all options, each category must meet its lower and upper quantity limits; category activation_fee_cents is deducted once if used. '
            'The incompatible table forbids joint selection, while requires means positive quantity of item_ref needs positive quantity of prerequisite_ref, without proportional quantities. '
            'Each bundle row contributes bonus_cents once if both options are selected; if either is unselected or unauthorized the bonus is zero. ')
    return s+'Report the maximum net benefit in USD cents. All fixed fees and bonuses are in the same unit.'

def main():
    backup=ROOT/'benchmark_archive/before_style_diversification'
    assert not backup.exists()
    backup.mkdir()
    shutil.copyfile(COPY,backup/'questions.csv')
    shutil.copyfile(BASE/'case_manifest.json',backup/'case_manifest.json')
    rows=core.read_csv(COPY);manifest=json.loads((BASE/'case_manifest.json').read_text())
    stage=BASE/'_style_stage';stage.mkdir()
    report=[]
    for i,(r,cfg) in enumerate(zip(rows,manifest)):
        name=cfg['case'];out=stage/name/'inputs';out.mkdir(parents=True)
        oldpaths=[ROOT/p for p in r['Dataset_address'].splitlines()]
        if i<3:
            records=[core.read_csv(p) for p in oldpaths]
            if name=='AP2':
                mat=records[2]
                records[2]=[{'worker_id':x['worker_id'],'project_id':p,'cost_cents':cost} for x in mat for p,cost in x.items() if p!='worker_id' and cost!='']
            elif name=='AP3': records=records[:2]+[records[2][j::3] for j in range(3)]
            paths=[]
            for j,data in enumerate(records):
                p=path_for(out,j);write(p,data);paths.append(p)
            t=v.ap_tables(paths);m,s,obj=core.solve_ap(t)
            intros={'AP1':'Assign six construction projects to the eight listed managers.',
                    'AP2':'A service team must cover eight jobs. The offer list gives cost_cents for each available worker_id/project_id pairing among ten staff members.',
                    'AP3':'Build a one-job-per-person roster for ten work packages using twelve available candidates. The cost rows are distributed across three files and must be matched using worker_id.'}
            r['Query']=intros[name]+' Each project needs exactly one person, and each person may take at most one project. Exclude anyone with on_leave=1. The assigned person must meet required_skill, with Junior < Intermediate < Senior < Expert. '
            r['Query']+=('Only listed offers are permitted. ' if name=='AP2' else 'In the cost matrix, project IDs are columns and worker IDs identify rows. A blank cell forbids that assignment. ')
            r['Query']+='Minimize the sum of assignment costs, and report the minimum in USD cents.'
            style='offer_list' if name=='AP2' else ('split_matrix' if name=='AP3' else 'matrix')
        else:
            original=v.ra_tables(oldpaths,cfg)
            t=json.loads(json.dumps(original));mode='events' if name in ['RA2','RA4'] else 'snapshot'
            if mode=='snapshot':
                if name not in ['RA4','RA13']:
                    fx={x['currency']:(int(x['usd_cents_numerator']),int(x['denominator'])) for x in t['fx']}
                    if name in ['RA1','RA3','RA12','RA14']:
                        vals=core.money(t.pop('benefit'),fx,'item_ref')
                        fees={x['item_ref']:x['activation_fee_cents'] for x in t.pop('item_fee')}
                        for x in t['item']:x.update(unit_benefit_cents=str(vals[x['item_ref']]),item_fee_cents=fees[x['item_ref']])
                    else:
                        for x in t['benefit']:
                            num,den=fx[x.pop('currency')];value=int(x.pop('amount'))*num
                            assert value%den==0;x['amount_cents']=str(value//den)
                    t.pop('fx')
                if name in ['RA3','RA7','RA14']:
                    scales={'liter':(1000,'ml'),'hour':(60,'minute'),'kwh':(1000,'wh'),'ml':(1,'ml'),'minute':(1,'minute'),'wh':(1,'wh')}
                    for tab in ['usage','capacity_ledger']:
                        for x in t[tab]:
                            factor,unit=scales[x['unit']];x['amount']=str(int(x['amount'])*factor);x['unit']=unit
                    # Compare with an equivalent base-unit original below.
                    for tab in ['usage','capacity_ledger']: original[tab]=json.loads(json.dumps(t[tab]))
                paths=[]
                for table,data in t.items():
                    parts=2 if name in ['RA7','RA10','RA14'] and table in ['item','usage'] else 1
                    for part in range(parts):
                        p=path_for(out,len(paths));write(p,[dict(table=table,**x) for x in data[part::parts]]);paths.append(p)
                cfg['input_style']='snapshot'
                for k in ['scope_column','scope_value','asof']:cfg.pop(k,None)
            else:
                paths=[]
                for oldpath in oldpaths:
                    data=core.read_csv(oldpath)
                    if name=='RA2':
                        if data[0]['table']=='fx':continue
                        for x in data:
                            if x['table']=='benefit':
                                val=x.pop('amount');currency=x.pop('currency');x['amount_cents']=str(int(val)//{'USD':1,'EUR':2,'JPY':100}.get(currency,1)) if val else ''
                            # Shift all dates together; effective business records do not change.
                            from datetime import date,timedelta
                            delta=date(2026,3,12)-date(2026,9,20)
                            x['effective_date']=(date.fromisoformat(x['effective_date'])+delta).isoformat()
                    p=path_for(out,len(paths));write(p,data);paths.append(p)
                cfg['input_style']='events'
                if name=='RA2':cfg['asof']='2026-03-12'
            restored=v.ra_tables(paths,cfg)
            assert economic_signature(restored)==economic_signature(original),name
            m,s=core.build_model(restored,'RA');obj=core.audit(restored,'RA',s)
            r['Query']=query(name,mode)
            style=mode+('_currency_components' if name in ['RA4','RA13'] else '_cents')
        assert obj==int(r['Label-objective']),(name,obj,r['Label-objective'])
        r['Dataset_address']='\n'.join(str((BASE/name/'inputs'/p.relative_to(out)).relative_to(ROOT)) for p in paths)
        report.append({'case':name,'style':style,'files':len(paths),'objective':obj})
    for cfg in manifest:
        name=cfg['case']
        shutil.move(str(BASE/name/'inputs'),str(backup/name))
        shutil.move(str(stage/name/'inputs'),str(BASE/name/'inputs'))
        (stage/name).rmdir()
    stage.rmdir()
    (BASE/'case_manifest.json').write_text(json.dumps(manifest,indent=2))
    write(BASE/'questions.csv',rows);COPY.write_bytes((BASE/'questions.csv').read_bytes())
    (BASE/'reference/style_diversification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
