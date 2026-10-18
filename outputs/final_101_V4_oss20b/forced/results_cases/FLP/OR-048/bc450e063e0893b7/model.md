##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\geq 0}$: number of units of air conditioner type $j$ placed in storage area $i$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}$

##### Constraints

1. Capacity constraints: $\displaystyle \sum_{j\in J} w_j\,x_{ij}\leq C_i,\quad \forall i\in I$  
2. Non‑negativity and integrality: $x_{ij}\in \mathbb{Z}_{\geq 0},\quad \forall i\in I,\;j\in J$  

$ I=\{1,2,\dots ,15\}$, $J=\{1,2,\dots ,10\}$.

##### Retrieved Information

```
{
  "storage_areas": [
    {"StorageID":"1","Capacity":"1083"},
    {"StorageID":"2","Capacity":"1840"},
    {"StorageID":"3","Capacity":"770"},
    {"StorageID":"4","Capacity":"1299"},
    {"StorageID":"5","Capacity":"1259"},
    {"StorageID":"6","Capacity":"543"},
    {"StorageID":"7","Capacity":"1831"},
    {"StorageID":"8","Capacity":"855"},
    {"StorageID":"9","Capacity":"619"},
    {"StorageID":"10","Capacity":"637"},
    {"StorageID":"11","Capacity":"935"},
    {"StorageID":"12","Capacity":"626"},
    {"StorageID":"13","Capacity":"1457"},
    {"StorageID":"14","Capacity":"1198"},
    {"StorageID":"15","Capacity":"837"}
  ],
  "products": [
    {"ProductName":"Window Unit","Value":"4811","Weight":"114"},
    {"ProductName":"Portable Unit","Value":"1130","Weight":"200"},
    {"ProductName":"Split System","Value":"1611","Weight":"106"},
    {"ProductName":"Ductless System","Value":"3368","Weight":"256"},
    {"ProductName":"Central AC","Value":"2135","Weight":"268"},
    {"ProductName":"Hybrid AC","Value":"1046","Weight":"185"},
    {"ProductName":"Geothermal AC","Value":"4030","Weight":"299"},
    {"ProductName":"Smart AC","Value":"3761","Weight":"131"},
    {"ProductName":"Evaporative Cooler","Value":"3523","Weight":"139"},
    {"ProductName":"Package Unit","Value":"1701","Weight":"105"}
  ],
  "capacity_vector": [1083,1840,770,1299,1259,543,1831,855,619,637,935,626,1457,1198,837],
  "value_vector": [4811,1130,1611,3368,2135,1046,4030,3761,3523,1701],
  "weight_vector": [114,200,106,256,268,185,299,131,139,105]
}
```