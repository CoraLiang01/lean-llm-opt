##### Sets and Indices

- Let $I$ be the set of storage areas, indexed by $i$, with StorageID as below.
- Let $J$ be the set of air conditioner types, indexed by $j$, with ProductName as below.

##### Parameters

- $C_i$: Capacity of storage area $i$ (from Capacity column, by StorageID)
- $v_j$: Value per unit of air conditioner type $j$ (from Value column, by ProductName)
- $w_j$: Size (Weight) per unit of air conditioner type $j$ (from Weight column, by ProductName)

##### Decision Variables

- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

##### Data

###### Storage Areas (from capacity.csv, in source order):

| StorageID | Capacity |
|-----------|----------|
| 1         | 1083     |
| 2         | 1840     |
| 3         | 770      |
| 4         | 1299     |
| 5         | 1259     |
| 6         | 543      |
| 7         | 1831     |
| 8         | 855      |
| 9         | 619      |
| 10        | 637      |
| 11        | 935      |
| 12        | 626      |
| 13        | 1457     |
| 14        | 1198     |
| 15        | 837      |

###### Air Conditioner Types (from products.csv, in source order):

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

##### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

- **Storage Area Capacity Constraints (for each $i$):**
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]

- **Integrality and Nonnegativity:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

##### Explicit Data Mapping

- $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$ (StorageID)
- $J = \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}$ (ProductName)
- $C_i$ as above for each $i$
- $v_j$, $w_j$ as above for each $j$

##### Decision Variables

- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$, integer and nonnegative.

---

**This is the complete integer optimization model as required by the problem and the retrieved data.**