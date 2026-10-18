##### Decision Variables

$x_{ij}\geq0$: number of vessels of type $j$ placed in display area $i$ (continuous).

##### Objective Function

$\max\sum_{i\in I}\sum_{j\in J}v_{j}\,x_{ij}$

##### Constraints

1. Capacity: $\sum_{j\in J}w_{j}\,x_{ij}\leq c_{i},\quad\forall i\in I$.
2. Non‑negativity: $x_{ij}\geq0,\quad\forall i\in I,\;\forall j\in J$.

$I=\{1,2,3,4,5,6,7,8,9,10,11,12,13,14\}$  
$J=\{\text{Speedboat},\text{Fishing Boat},\text{Catamaran},\text{Yacht},\text{Sailboat},\text{Kayak},\text{Canoe},\text{Houseboat},\text{Pontoon},\text{Jet Ski},\text{Rowboat},\text{Hovercraft},\text{Cabin Cruiser},\text{Wakeboard Boat},\text{Dinghy},\text{Trawler},\text{Paddle Boat},\text{Submarine},\text{RIB},\text{Skiff}\}$  

$c_{i}$ and $w_{j}$ are the capacity of display $i$ and the weight of vessel $j$, respectively; $v_{j}$ is the value of vessel $j$.

###### Retrieved Information

{
  "displays": ["1","2","3","4","5","6","7","8","9","10","11","12","13","14"],
  "capacity": {
    "1":457,
    "2":604,
    "3":751,
    "4":468,
    "5":343,
    "6":408,
    "7":741,
    "8":914,
    "9":682,
    "10":409,
    "11":342,
    "12":903,
    "13":680,
    "14":886
  },
  "products": [
    "Speedboat",
    "Fishing Boat",
    "Catamaran",
    "Yacht",
    "Sailboat",
    "Kayak",
    "Canoe",
    "Houseboat",
    "Pontoon",
    "Jet Ski",
    "Rowboat",
    "Hovercraft",
    "Cabin Cruiser",
    "Wakeboard Boat",
    "Dinghy",
    "Trawler",
    "Paddle Boat",
    "Submarine",
    "RIB",
    "Skiff"
  ],
  "value": {
    "Speedboat":29664,
    "Fishing Boat":31778,
    "Catamaran":73501,
    "Yacht":78255,
    "Sailboat":93606,
    "Kayak":46983,
    "Canoe":95026,
    "Houseboat":57685,
    "Pontoon":60323,
    "Jet Ski":91224,
    "Rowboat":44003,
    "Hovercraft":75998,
    "Cabin Cruiser":84525,
    "Wakeboard Boat":66207,
    "Dinghy":65002,
    "Trawler":33132,
    "Paddle Boat":69239,
    "Submarine":66948,
    "RIB":88240,
    "Skiff":48858
  },
  "weight": {
    "Speedboat":18,
    "Fishing Boat":36,
    "Catamaran":25,
    "Yacht":16,
    "Sailboat":97,
    "Kayak":35,
    "Canoe":32,
    "Houseboat":100,
    "Pontoon":43,
    "Jet Ski":15,
    "Rowboat":95,
    "Hovercraft":57,
    "Cabin Cruiser":13,
    "Wakeboard Boat":44,
    "Dinghy":64,
    "Trawler":88,
    "Paddle Boat":42,
    "Submarine":46,
    "RIB":24,
    "Skiff":93
  }
}