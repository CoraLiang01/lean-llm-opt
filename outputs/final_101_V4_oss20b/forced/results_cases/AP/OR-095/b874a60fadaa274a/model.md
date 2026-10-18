##### Objective Function: 

$ \quad \max \sum_{i=1}^{141} \text{Profit}_{i}\,x_{i} + 300\,y - 200\,z $

##### Constraints

###### 1. Labor Hours Constraint: 

$ \sum_{i=1}^{141} \text{LaborHours}_{i}\,x_{i} \le 5000 $

###### 2. Material A Constraint: 

$ \sum_{i=1}^{141} \text{MaterialA}_{i}\,x_{i} \le 24000 $

###### 3. Material B Constraint: 

$ \sum_{i=1}^{141} \text{MaterialB}_{i}\,x_{i} \le 15000 $

###### 4. CatalystX Production and Sale Constraints: 

$ y \le 1500 $  

$ y \le 5\,x_{3} $  

$ z = 5\,x_{3} - y $

###### 5. Non‑negativity Constraints: 

$ x_{i} \ge 0 \quad \forall i=1,\dots ,141 $  

$ y \ge 0 $  

$ z \ge 0 $

###### 6. Variable Type Constraints: 

$ x_{i},\,y,\,z \in \mathbb{R} $

##### Retrieved Information

{
  "widgets": [
    {"name":"Widget1","LaborHours":1.6,"MaterialA":24,"MaterialB":14,"Profit":525},
    {"name":"Widget2","LaborHours":2.0,"MaterialA":20,"MaterialB":10,"Profit":678},
    {"name":"Widget3","LaborHours":2.5,"MaterialA":12,"MaterialB":18,"Profit":812},
    {"name":"Widget4","LaborHours":1.9,"MaterialA":21,"MaterialB":15,"Profit":769},
    {"name":"Widget5","LaborHours":0.0,"MaterialA":15,"MaterialB":26,"Profit":952},
    {"name":"Widget6","LaborHours":0.1,"MaterialA":24,"MaterialB":17,"Profit":987},
    {"name":"Widget7","LaborHours":1.2,"MaterialA":15,"MaterialB":30,"Profit":644},
    {"name":"Widget8","LaborHours":1.3,"MaterialA":21,"MaterialB":24,"Profit":795},
    {"name":"Widget9","LaborHours":0.4,"MaterialA":20,"MaterialB":30,"Profit":829},
    {"name":"Widget10","LaborHours":0.9,"MaterialA":18,"MaterialB":27,"Profit":574},
    {"name":"Widget11","LaborHours":0.2,"MaterialA":24,"MaterialB":21,"Profit":723},
    {"name":"Widget12","LaborHours":1.6,"MaterialA":20,"MaterialB":20,"Profit":775},
    {"name":"Widget13","LaborHours":0.1,"MaterialA":20,"MaterialB":22,"Profit":992},
    {"name":"Widget14","LaborHours":0.7,"MaterialA":12,"MaterialB":19,"Profit":568},
    {"name":"Widget15","LaborHours":1.1,"MaterialA":11,"MaterialB":28,"Profit":895},
    {"name":"Widget16","LaborHours":0.7,"MaterialA":15,"MaterialB":25,"Profit":838},
    {"name":"Widget17","LaborHours":1.0,"MaterialA":14,"MaterialB":27,"Profit":883},
    {"name":"Widget18","LaborHours":1.8,"MaterialA":18,"MaterialB":15,"Profit":567},
    {"name":"Widget19","LaborHours":0.5,"MaterialA":23,"MaterialB":25,"Profit":829},
    {"name":"Widget20","LaborHours":1.2,"MaterialA":17,"MaterialB":29,"Profit":821},
    {"name":"Widget21","LaborHours":0.2,"MaterialA":20,"MaterialB":17,"Profit":893},
    {"name":"Widget22","LaborHours":1.5,"MaterialA":10,"MaterialB":22,"Profit":977},
    {"name":"Widget23","LaborHours":1.1,"MaterialA":12,"MaterialB":21,"Profit":536},
    {"name":"Widget24","LaborHours":0.6,"MaterialA":12,"MaterialB":24,"Profit":797},
    {"name":"Widget25","LaborHours":1.8,"MaterialA":10,"MaterialB":25,"Profit":793},
    {"name":"Widget26","LaborHours":2.0,"MaterialA":15,"MaterialB":23,"Profit":754},
    {"name":"Widget27","LaborHours":1.4,"MaterialA":17,"MaterialB":21,"Profit":819},
    {"name":"Widget28","LaborHours":1.3,"MaterialA":22,"MaterialB":30,"Profit":947},
    {"name":"Widget29","LaborHours":1.2,"MaterialA":11,"MaterialB":17,"Profit":857},
    {"name":"Widget30","LaborHours":0.3,"MaterialA":13,"MaterialB":29,"Profit":886},
    {"name":"Widget31","LaborHours":1.6,"MaterialA":17,"MaterialB":15,"Profit":609},
    {"name":"Widget32","LaborHours":0.9,"MaterialA":24,"MaterialB":19,"Profit":706},
    {"name":"Widget33","LaborHours":0.1,"MaterialA":10,"MaterialB":26,"Profit":545},
    {"name":"Widget34","LaborHours":0.0,"MaterialA":12,"MaterialB":22,"Profit":762},
    {"name":"Widget35","LaborHours":1.8,"MaterialA":12,"MaterialB":21,"Profit":719},
    {"name":"Widget36","LaborHours":0.3,"MaterialA":24,"MaterialB":26,"Profit":793},
    {"name":"Widget37","LaborHours":0.6,"MaterialA":20,"MaterialB":27,"Profit":865},
    {"name":"Widget38","LaborHours":0.7,"MaterialA":16,"MaterialB":23,"Profit":932},
    {"name":"Widget39","LaborHours":0.6,"MaterialA":17,"MaterialB":18,"Profit":962},
    {"name":"Widget40","LaborHours":1.3,"MaterialA":10,"MaterialB":22,"Profit":824},
    {"name":"Widget41","LaborHours":0.8,"MaterialA":15,"MaterialB":19,"Profit":992},
    {"name":"Widget42","LaborHours":1.7,"MaterialA":21,"MaterialB":29,"Profit":536},
    {"name":"Widget43","LaborHours":1.5,"MaterialA":17,"MaterialB":25,"Profit":896},
    {"name":"Widget44","LaborHours":1.3,"MaterialA":25,"MaterialB":23,"Profit":969},
    {"name":"Widget45","LaborHours":1.8,"MaterialA":21,"MaterialB":26,"Profit":558},
    {"name":"Widget46","LaborHours":0.1,"MaterialA":25,"MaterialB":20,"Profit":893},
    {"name":"Widget47","LaborHours":1.0,"MaterialA":11,"MaterialB":30,"Profit":777},
    {"name":"Widget48","LaborHours":0.6,"MaterialA":23,"MaterialB":29,"Profit":718},
    {"name":"Widget49","LaborHours":0.2,"MaterialA":25,"MaterialB":16,"Profit":811},
    {"name":"Widget50","LaborHours":1.2,"MaterialA":25,"MaterialB":29,"Profit":545},
    {"name":"Widget51","LaborHours":0.7,"MaterialA":17,"MaterialB":25,"Profit":961},
    {"name":"Widget52","LaborHours":1.7,"MaterialA":12,"MaterialB":25,"Profit":657},
    {"name":"Widget53","LaborHours":0.8,"MaterialA":12,"MaterialB":23,"Profit":658},
    {"name":"Widget54","LaborHours":1.6,"MaterialA":15,"MaterialB":19,"Profit":860},
    {"name":"Widget55","LaborHours":0.5,"MaterialA":22,"MaterialB":20,"Profit":625},
    {"name":"Widget56","LaborHours":1.1,"MaterialA":18,"MaterialB":17,"Profit":510},
    {"name":"Widget57","LaborHours":0.0,"MaterialA":17,"MaterialB":24,"Profit":1000},
    {"name":"Widget58","LaborHours":0.2,"MaterialA":22,"MaterialB":17,"Profit":544},
    {"name":"Widget59","LaborHours":1.7,"MaterialA":23,"MaterialB":17,"Profit":547},
    {"name":"Widget60","LaborHours":0.5,"MaterialA":18,"MaterialB":27,"Profit":961},
    {"name":"Widget61","LaborHours":1.6,"MaterialA":16,"MaterialB":23,"Profit":689},
    {"name":"Widget62","LaborHours":1.1,"MaterialA":12,"MaterialB":16,"Profit":545},
    {"name":"Widget63","LaborHours":0.5,"MaterialA":13,"MaterialB":16,"Profit":897},
    {"name":"Widget64","LaborHours":0.1,"MaterialA":13,"MaterialB":29,"Profit":881},
    {"name":"Widget65","LaborHours":0.6,"MaterialA":25,"MaterialB":24,"Profit":972},
    {"name":"Widget66","LaborHours":1.9,"MaterialA":18,"MaterialB":17,"Profit":860},
    {"name":"Widget67","LaborHours":0.7,"MaterialA":23,"MaterialB":15,"Profit":881},
    {"name":"Widget68","LaborHours":1.3,"MaterialA":25,"MaterialB":15,"Profit":892},
    {"name":"Widget69","LaborHours":1.5,"MaterialA":22,"MaterialB":27,"Profit":797},
    {"name":"Widget70","LaborHours":0.4,"MaterialA":14,"MaterialB":21,"Profit":767},
    {"name":"Widget71","LaborHours":1.1,"MaterialA":14,"MaterialB":22,"Profit":794},
    {"name":"Widget72","LaborHours":0.3,"MaterialA":15,"MaterialB":30,"Profit":899},
    {"name":"Widget73","LaborHours":0.6,"MaterialA":10,"MaterialB":26,"Profit":504},
    {"name":"Widget74","LaborHours":1.5,"MaterialA":10,"MaterialB":26,"Profit":731},
    {"name":"Widget75","LaborHours":0.2,"MaterialA":23,"MaterialB":19,"Profit":929},
    {"name":"Widget76","LaborHours":0.3,"MaterialA":22,"MaterialB":19,"Profit":872},
    {"name":"Widget77","LaborHours":1.0,"MaterialA":14,"MaterialB":25,"Profit":631},
    {"name":"Widget78","LaborHours":1.3,"MaterialA":14,"MaterialB":21,"Profit":681},
    {"name":"Widget79","LaborHours":2.0,"MaterialA":22,"MaterialB":28,"Profit":587},
    {"name":"Widget80","LaborHours":1.8,"MaterialA":22,"MaterialB":21,"Profit":829},
    {"name":"Widget81","LaborHours":1.2,"MaterialA":12,"MaterialB":29,"Profit":734},
    {"name":"Widget82","LaborHours":1.1,"MaterialA":12,"MaterialB":19,"Profit":648},
    {"name":"Widget83","LaborHours":0.3,"MaterialA":22,"MaterialB":26,"Profit":738},
    {"name":"Widget84","LaborHours":1.4,"MaterialA":23,"MaterialB":19,"Profit":648},
    {"name":"Widget85","LaborHours":1.1,"MaterialA":11,"MaterialB":30,"Profit":557},
    {"name":"Widget86","LaborHours":1.6,"MaterialA":23,"MaterialB":28,"Profit":960},
    {"name":"Widget87","LaborHours":1.6,"MaterialA":14,"MaterialB":15,"Profit":907},
    {"name":"Widget88","LaborHours":1.5,"MaterialA":17,"MaterialB":24,"Profit":531},
    {"name":"Widget89","LaborHours":0.0,"MaterialA":23,"MaterialB":21,"Profit":887},
    {"name":"Widget90","LaborHours":0.6,"MaterialA":13,"MaterialB":21,"Profit":675},
    {"name":"Widget91","LaborHours":0.2,"MaterialA":25,"MaterialB":20,"Profit":766},
    {"name":"Widget92","LaborHours":0.5,"MaterialA":15,"MaterialB":18,"Profit":861},
    {"name":"Widget93","LaborHours":1.4,"MaterialA":24,"MaterialB":20,"Profit":841},
    {"name":"Widget94","LaborHours":1.6,"MaterialA":17,"MaterialB":28,"Profit":618},
    {"name":"Widget95","LaborHours":1.8,"MaterialA":23,"MaterialB":17,"Profit":574},
    {"name":"Widget96","LaborHours":0.5,"MaterialA":13,"MaterialB":18,"Profit":748},
    {"name":"Widget97","LaborHours":1.6,"MaterialA":20,"MaterialB":21,"Profit":898},
    {"name":"Widget98","LaborHours":1.4,"MaterialA":16,"MaterialB":18,"Profit":520},
    {"name":"Widget99","LaborHours":0.3,"MaterialA":11,"MaterialB":19,"Profit":675},
    {"name":"Widget100","LaborHours":1.3,"MaterialA":22,"MaterialB":26,"Profit":868},
    {"name":"Widget101","LaborHours":1.5,"MaterialA":18,"MaterialB":19,"Profit":698},
    {"name":"Widget102","LaborHours":0.7,"MaterialA":12,"MaterialB":17,"Profit":988},
    {"name":"Widget103","LaborHours":0.8,"MaterialA":12,"MaterialB":29,"Profit":550},
    {"name":"Widget104","LaborHours":1.8,"MaterialA":10,"MaterialB":27,"Profit":508},
    {"name":"Widget105","LaborHours":1.7,"MaterialA":24,"MaterialB":22,"Profit":682},
    {"name":"Widget106","LaborHours":0.8,"MaterialA":21,"MaterialB":20,"Profit":995},
    {"name":"Widget107","LaborHours":0.3,"MaterialA":25,"MaterialB":30,"Profit":656},
    {"name":"Widget108","LaborHours":1.3,"MaterialA":25,"MaterialB":22,"Profit":569},
    {"name":"Widget109","LaborHours":0.9,"MaterialA":21,"MaterialB":18,"Profit":779},
    {"name":"Widget110","LaborHours":1.6,"MaterialA":23,"MaterialB":26,"Profit":965},
    {"name":"Widget111","LaborHours":0.4,"MaterialA":23,"MaterialB":27,"Profit":675},
    {"name":"Widget112","LaborHours":1.7,"MaterialA":21,"MaterialB":30,"Profit":638},
    {"name":"Widget113","LaborHours":1.5,"MaterialA":20,"MaterialB":26,"Profit":888},
    {"name":"Widget114","LaborHours":2.0,"MaterialA":15,"MaterialB":26,"Profit":875},
    {"name":"Widget115","LaborHours":0.9,"MaterialA":24,"MaterialB":24,"Profit":552},
    {"name":"Widget116","LaborHours":0.9,"MaterialA":12,"MaterialB":23,"Profit":705},
    {"name":"Widget117","LaborHours":0.1,"MaterialA":15,"MaterialB":28,"Profit":863},
    {"name":"Widget118","LaborHours":0.1,"MaterialA":14,"MaterialB":18,"Profit":706},
    {"name":"Widget119","LaborHours":0.3,"MaterialA":13,"MaterialB":25,"Profit":663},
    {"name":"Widget120","LaborHours":0.1,"MaterialA":13,"MaterialB":20,"Profit":859},
    {"name":"Widget121","LaborHours":1.3,"MaterialA":24,"MaterialB":17,"Profit":809},
    {"name":"Widget122","LaborHours":1.5,"MaterialA":12,"MaterialB":17,"Profit":830},
    {"name":"Widget123","LaborHours":1.7,"MaterialA":19,"MaterialB":29,"Profit":991},
    {"name":"Widget124","LaborHours":0.7,"MaterialA":10,"MaterialB":16,"Profit":791},
    {"name":"Widget125","LaborHours":0.6,"MaterialA":13,"MaterialB":22,"Profit":934},
    {"name":"Widget126","LaborHours":0.8,"MaterialA":22,"MaterialB":15,"Profit":732},
    {"name":"Widget127","LaborHours":0.4,"MaterialA":22,"MaterialB":17,"Profit":989},
    {"name":"Widget128","LaborHours":1.4,"MaterialA":21,"MaterialB":21,"Profit":573},
    {"name":"Widget129","LaborHours":0.1,"MaterialA":16,"MaterialB":23,"Profit":720},
    {"name":"Widget130","LaborHours":0.5,"MaterialA":19,"MaterialB":27,"Profit":834},
    {"name":"Widget131","LaborHours":0.1,"MaterialA":23,"MaterialB":18,"Profit":577},
    {"name":"Widget132","LaborHours":0.9,"MaterialA":17,"MaterialB":20,"Profit":951},
    {"name":"Widget133","LaborHours":1.1,"MaterialA":19,"MaterialB":30,"Profit":882},
    {"name":"Widget134","LaborHours":1.7,"MaterialA":17,"MaterialB":17,"Profit":859},
    {"name":"Widget135","LaborHours":1.3,"MaterialA":22,"MaterialB":17,"Profit":568},
    {"name":"Widget136","LaborHours":0.8,"MaterialA":17,"MaterialB":26,"Profit":861},
    {"name":"Widget137","LaborHours":1.7,"MaterialA":13,"MaterialB":17,"Profit":875},
    {"name":"Widget138","LaborHours":1.7,"MaterialA":12,"MaterialB":18,"Profit":834},
    {"name":"Widget139","LaborHours":1.6,"MaterialA":12,"MaterialB":23,"Profit":836},
    {"name":"Widget140","LaborHours":1.6,"MaterialA":17,"MaterialB":24,"Profit":938},
    {"name":"Widget141","LaborHours":1.2,"MaterialA":11,"MaterialB":16,"Profit":593}
  ],
  "resource_limits": {
    "LaborHours": 5000,
    "MaterialA": 24000,
    "MaterialB": 15000
  },
  "catalyst": {
    "product": "Widget3",
    "rate_per_unit": 5,
    "sale_price_per_kg": 300,
    "disposal_cost_per_kg": 200,
    "sale_cap_kg": 1500
  }
}