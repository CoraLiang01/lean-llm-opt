Let:
- $i$ index the storage areas (with StorageID from capacity.csv)
- $j$ index the air conditioner types (with ProductName from products.csv)
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$ (integer, $\geq 0$)

Parameters:
- $v_j$ = Value of air conditioner type $j$ (from products.csv)
- $w_j$ = Weight (size) of air conditioner type $j$ (from products.csv)
- $C_i$ = Capacity of storage area $i$ (from capacity.csv)

Sets (in source order):

Storage Areas (StorageID, Capacity):
1: 1083  
2: 1840  
3: 770  
4: 1299  
5: 1259  
6: 543  
7: 1831  
8: 855  
9: 619  
10: 637  
11: 935  
12: 626  
13: 1457  
14: 1198  
15: 837  

Air Conditioner Types (ProductName, Value, Weight):
- Window Unit: 4811, 114
- Portable Unit: 1130, 200
- Split System: 1611, 106
- Ductless System: 3368, 256
- Central AC: 2135, 268
- Hybrid AC: 1046, 185
- Geothermal AC: 4030, 299
- Smart AC: 3761, 131
- Evaporative Cooler: 3523, 139
- Package Unit: 1701, 105

Model:

Objective:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

Subject to (for each storage area $i$):

Capacity constraints:
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

Where:

- $C_i$ is as listed above for each StorageID.
- $v_j$ and $w_j$ are as listed above for each ProductName.

All other columns in the data are not referenced by the user query and are not used in the model.