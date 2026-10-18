##### Decision Variables

$x_{ij}\in\mathbb{Z}_{\ge 0}$: number of units of air‑conditioner type $j$ placed in storage area $i$.

##### Objective Function

$\displaystyle \max\sum_{i\in I}\sum_{j\in J}v_{j}\,x_{ij}$

##### Constraints

1. Capacity: $\displaystyle \sum_{j\in J}a_{j}\,x_{ij}\le s_{i},\quad\forall i\in I$  
2. Non‑negativity and integrality: $x_{ij}\in\mathbb{Z}_{\ge 0},\quad\forall i\in I,\;j\in J$  

$ I$ = set of storage areas (from **capacity.csv**).  
$ J$ = set of air‑conditioner types (from **products.csv**).  
$s_{i}$ = capacity of storage area $i$.  
$a_{j}$ = size (weight) of one unit of type $j$.  
$v_{j}$ = value of one unit of type $j$.

###### Retrieved Information

```json
{
  "storage_areas": ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"],
  "capacities": {
    "1":1083,
    "2":1840,
    "3":770,
    "4":1299,
    "5":1259,
    "6":543,
    "7":1831,
    "8":855,
    "9":619,
    "10":637,
    "11":935,
    "12":626,
    "13":1457,
    "14":1198,
    "15":837
  },
  "products": [
    "Window Unit",
    "Portable Unit",
    "Split System",
    "Ductless System",
    "Central AC",
    "Hybrid AC",
    "Geothermal AC",
    "Smart AC",
    "Evaporative Cooler",
    "Package Unit"
  ],
  "sizes": {
    "Window Unit":114,
    "Portable Unit":200,
    "Split System":106,
    "Ductless System":256,
    "Central AC":268,
    "Hybrid AC":185,
    "Geothermal AC":299,
    "Smart AC":131,
    "Evaporative Cooler":139,
    "Package Unit":105
  },
  "values": {
    "Window Unit":4811,
    "Portable Unit":1130,
    "Split System":1611,
    "Ductless System":3368,
    "Central AC":2135,
    "Hybrid AC":1046,
    "Geothermal AC":4030,
    "Smart AC":3761,
    "Evaporative Cooler":3523,
    "Package Unit":1701
  }
}
```