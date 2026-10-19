##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\ge 0}$: number of units of boat type $j$ placed in display area $i$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J} v_{j}\,x_{ij}$

##### Constraints

1. Capacity constraints: $\displaystyle \sum_{j\in J} w_{j}\,x_{ij}\le C_{i},\quad \forall i\in I$.

2. Domains: $x_{ij}\in \mathbb{Z}_{\ge 0}$.

##### Sets

$I=\{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$,  
$J=\{\text{Speedboat},\text{Fishing Boat},\text{Catamaran},\text{Yacht},\text{Sailboat},\text{Kayak},\text{Canoe},\text{Houseboat},\text{Pontoon},\text{Jet Ski},\text{Rowboat},\text{Hovercraft},\text{Cabin Cruiser},\text{Wakeboard Boat},\text{Dinghy},\text{Trawler},\text{Paddle Boat},\text{Submarine},\text{RIB},\text{Skiff}\}$.

##### Retrieved Information

{
  "capacities": [
    {"DisplayID":"1","Capacity":356},
    {"DisplayID":"2","Capacity":478},
    {"DisplayID":"3","Capacity":305},
    {"DisplayID":"4","Capacity":291},
    {"DisplayID":"5","Capacity":168},
    {"DisplayID":"6","Capacity":449},
    {"DisplayID":"7","Capacity":139},
    {"DisplayID":"8","Capacity":383},
    {"DisplayID":"9","Capacity":472},
    {"DisplayID":"10","Capacity":288},
    {"DisplayID":"11","Capacity":320},
    {"DisplayID":"12","Capacity":250},
    {"DisplayID":"13","Capacity":402},
    {"DisplayID":"14","Capacity":293}
  ],
  "products": [
    {"ProductName":"Speedboat","Value":69978,"Weight":18},
    {"ProductName":"Fishing Boat","Value":54011,"Weight":42},
    {"ProductName":"Catamaran","Value":36352,"Weight":49},
    {"ProductName":"Yacht","Value":51521,"Weight":42},
    {"ProductName":"Sailboat","Value":50415,"Weight":41},
    {"ProductName":"Kayak","Value":76109,"Weight":48},
    {"ProductName":"Canoe","Value":50462,"Weight":22},
    {"ProductName":"Houseboat","Value":28989,"Weight":29},
    {"ProductName":"Pontoon","Value":23318,"Weight":45},
    {"ProductName":"Jet Ski","Value":26142,"Weight":14},
    {"ProductName":"Rowboat","Value":42040,"Weight":38},
    {"ProductName":"Hovercraft","Value":85961,"Weight":47},
    {"ProductName":"Cabin Cruiser","Value":50142,"Weight":45},
    {"ProductName":"Wakeboard Boat","Value":48478,"Weight":28},
    {"ProductName":"Dinghy","Value":60953,"Weight":24},
    {"ProductName":"Trawler","Value":95265,"Weight":39},
    {"ProductName":"Paddle Boat","Value":22839,"Weight":32},
    {"ProductName":"Submarine","Value":90957,"Weight":36},
    {"ProductName":"RIB","Value":84652,"Weight":14},
    {"ProductName":"Skiff","Value":78991,"Weight":16}
  ]
}