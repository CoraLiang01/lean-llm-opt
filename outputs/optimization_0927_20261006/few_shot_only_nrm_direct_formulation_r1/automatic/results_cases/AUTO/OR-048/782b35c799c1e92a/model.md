**Sets:**
- Let $I$ be the set of storage areas (indexed by $i$), with StorageID from capacity.csv.
- Let $J$ be the set of air conditioner types (indexed by $j$), with ProductName from products.csv.

**Parameters:**
- $C_i$: Capacity of storage area $i$ (from capacity.csv).
- $v_j$: Value of one unit of product $j$ (from products.csv).
- $w_j$: Weight (size) of one unit of product $j$ (from products.csv).

**Decision Variables:**
- $x_{ij}$: Number of units of product $j$ placed in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

For each storage area $i$:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

**Numerical Formulation with Retrieved Data:**

Let $I = \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$ (StorageID from capacity.csv).

Let $J = \{$
- Window Unit,
- Portable Unit,
- Split System,
- Ductless System,
- Central AC,
- Hybrid AC,
- Geothermal AC,
- Smart AC,
- Evaporative Cooler,
- Package Unit
$\}$ (ProductName from products.csv).

Parameters:

From capacity.csv:
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

From products.csv:

| ProductName           | $v_j$ (Value) | $w_j$ (Weight) |
|-----------------------|---------------|---------------|
| Window Unit           | 4811          | 114           |
| Portable Unit         | 1130          | 200           |
| Split System          | 1611          | 106           |
| Ductless System       | 3368          | 256           |
| Central AC            | 2135          | 268           |
| Hybrid AC             | 1046          | 185           |
| Geothermal AC         | 4030          | 299           |
| Smart AC              | 3761          | 131           |
| Evaporative Cooler    | 3523          | 139           |
| Package Unit          | 1701          | 105           |

**Model:**

\[
\max \sum_{i=1}^{15} \Bigg[
4811\,x_{i,\text{Window Unit}} + 1130\,x_{i,\text{Portable Unit}} + 1611\,x_{i,\text{Split System}} + 3368\,x_{i,\text{Ductless System}} + 2135\,x_{i,\text{Central AC}} + 1046\,x_{i,\text{Hybrid AC}} + 4030\,x_{i,\text{Geothermal AC}} + 3761\,x_{i,\text{Smart AC}} + 3523\,x_{i,\text{Evaporative Cooler}} + 1701\,x_{i,\text{Package Unit}}
\Bigg]
\]

Subject to, for each $i = 1,\ldots,15$:

\[
114\,x_{i,\text{Window Unit}} + 200\,x_{i,\text{Portable Unit}} + 106\,x_{i,\text{Split System}} + 256\,x_{i,\text{Ductless System}} + 268\,x_{i,\text{Central AC}} + 185\,x_{i,\text{Hybrid AC}} + 299\,x_{i,\text{Geothermal AC}} + 131\,x_{i,\text{Smart AC}} + 139\,x_{i,\text{Evaporative Cooler}} + 105\,x_{i,\text{Package Unit}} \leq C_i
\]

where $C_i$ is as listed above for each StorageID.

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\, j \in J
\]