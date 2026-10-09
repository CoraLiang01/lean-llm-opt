Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

  with capacities:
  - $C_1 = 1083$
  - $C_2 = 1840$
  - $C_3 = 770$
  - $C_4 = 1299$
  - $C_5 = 1259$
  - $C_6 = 543$
  - $C_7 = 1831$
  - $C_8 = 855$
  - $C_9 = 619$
  - $C_{10} = 637$
  - $C_{11} = 935$
  - $C_{12} = 626$
  - $C_{13} = 1457$
  - $C_{14} = 1198$
  - $C_{15} = 837$

- Air conditioner types (from products.csv):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Window Unit           | 4811  | 114    |
| Portable Unit         | 1130  | 200    |
| Split System          | 1611  | 106    |
| Ductless System       | 3368  | 256    |
| Central AC            | 2135  | 268    |
| Hybrid AC             | 1046  | 185    |
| Geothermal AC         | 4030  | 299    |
| Smart AC              | 3761  | 131    |
| Evaporative Cooler    | 3523  | 139    |
| Package Unit          | 1701  | 105    |

**Mathematical Model:**

Maximize total value:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value for product $j$ as given above.

Subject to, for each storage area $i$:

$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

where $w_j$ is the Weight for product $j$ as given above, and $C_i$ is the Capacity for storage area $i$.

And

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Explicitly:**

Let the set of storage areas $S = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$,  
and the set of air conditioner types $P = \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$.

For all $i \in S$, $j \in P$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$

For all $i \in S$:

$$
114\,x_{i,\text{Window Unit}} + 200\,x_{i,\text{Portable Unit}} + 106\,x_{i,\text{Split System}} + 256\,x_{i,\text{Ductless System}} + 268\,x_{i,\text{Central AC}} + 185\,x_{i,\text{Hybrid AC}} + 299\,x_{i,\text{Geothermal AC}} + 131\,x_{i,\text{Smart AC}} + 139\,x_{i,\text{Evaporative Cooler}} + 105\,x_{i,\text{Package Unit}} \leq C_i
$$

where $C_i$ is as listed above for each StorageID.

**Objective:**

$$
\max \sum_{i \in S} \Big[
4811\,x_{i,\text{Window Unit}} +
1130\,x_{i,\text{Portable Unit}} +
1611\,x_{i,\text{Split System}} +
3368\,x_{i,\text{Ductless System}} +
2135\,x_{i,\text{Central AC}} +
1046\,x_{i,\text{Hybrid AC}} +
4030\,x_{i,\text{Geothermal AC}} +
3761\,x_{i,\text{Smart AC}} +
3523\,x_{i,\text{Evaporative Cooler}} +
1701\,x_{i,\text{Package Unit}}
\Big]
$$

**Decision variables:**  
$x_{ij}$: integer, $\geq 0$, for all storage areas $i$ and product types $j$.

**All data and constraints are included as retrieved and described.**