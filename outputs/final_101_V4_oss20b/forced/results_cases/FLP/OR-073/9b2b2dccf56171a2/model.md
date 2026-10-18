##### Decision Variables

$x_{p,e}\geq0$: quantity of product $p\in P$ processed on equipment $e\in E$ (continuous).  
$y_e\in\{0,1\}$: whether equipment $e$ is activated.

##### Objective Function

$\displaystyle \max \sum_{p\in P}\sum_{e\in E}(price_p-raw_p)\,x_{p,e}\;-\;\sum_{e\in E}cost_e\,y_e$

##### Constraints

1. Equipment time:  
$\displaystyle \sum_{p\in P}time_{e,p}\,x_{p,e}\;\leq\;available_e\,y_e,\quad \forall e\in E$.

2. Product eligibility:  
$\displaystyle x_{I,A3}=0,$  
$\displaystyle x_{III,A1}=0,$  
$\displaystyle x_{III,A3}=0,$  
$\displaystyle x_{III,B1}=0,$  
$\displaystyle x_{III,B3}=0,$  
$\displaystyle x_{III,B4}=0,$  
$\displaystyle x_{II,B2}=0,$  
$\displaystyle x_{II,B3}=0,$  
$\displaystyle x_{II,B4}=0.$

3. Domains:  
$\displaystyle x_{p,e}\geq0,\quad y_e\in\{0,1\}$.

$P=\{I,II,III\}$, $E=\{A1,A2,A3,B1,B2,B3,B4\}$.

###### Retrieved Information

{
  "products": ["I","II","III"],
  "equipment": ["A1","A2","A3","B1","B2","B3","B4"],
  "processing_time": {
    "A1": {"I":5,"II":10,"III":0},
    "A2": {"I":7,"II":9,"III":12},
    "A3": {"I":6,"II":11,"III":2},
    "B1": {"I":6,"II":8,"III":0},
    "B2": {"I":4,"II":0,"III":11},
    "B3": {"I":7,"II":0,"III":0},
    "B4": {"I":3,"II":5,"III":8}
  },
  "available_time": {"A1":6000,"A2":10000,"A3":8000,"B1":4000,"B2":7000,"B3":4000,"B4":5000},
  "equipment_cost": {"A1":300,"A2":321,"A3":203,"B1":250,"B2":783,"B3":200,"B4":300},
  "raw_material_cost": {"I":0.25,"II":0.35,"III":0.5},
  "unit_price": {"I":1.25,"II":2,"III":2.8}
}