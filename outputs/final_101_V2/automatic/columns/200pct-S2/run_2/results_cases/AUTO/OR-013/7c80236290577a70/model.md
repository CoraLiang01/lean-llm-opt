Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv, in source order):  
  $i \in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$, with capacities:
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

- Air conditioner types (from products.csv, in source order):  
  $j \in \{$Window Unit, Portable Unit, Split System, Ductless System, Central AC, Hybrid AC, Geothermal AC, Smart AC, Evaporative Cooler, Package Unit$\}$, with:
    - Value $v_j$:
        - Window Unit: $4811$
        - Portable Unit: $1130$
        - Split System: $1611$
        - Ductless System: $3368$
        - Central AC: $2135$
        - Hybrid AC: $1046$
        - Geothermal AC: $4030$
        - Smart AC: $3761$
        - Evaporative Cooler: $3523$
        - Package Unit: $1701$
    - Size $w_j$:
        - Window Unit: $114$
        - Portable Unit: $200$
        - Split System: $106$
        - Ductless System: $256$
        - Central AC: $268$
        - Hybrid AC: $185$
        - Geothermal AC: $299$
        - Smart AC: $131$
        - Evaporative Cooler: $139$
        - Package Unit: $105$

---

**Mathematical Model**

**Decision variables:**  
$x_{ij} \in \mathbb{Z}_{\geq 0}$, for all storage areas $i$ and air conditioner types $j$.

**Objective:**  
Maximize the total value of air conditioners allocated:
$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \ldots, \text{Package Unit}\}} v_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i$ (with capacity $C_i$):
$$
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,15\}
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Where:**

- $v_j$ and $w_j$ are as listed above for each ProductName $j$.
- $C_i$ is as listed above for each StorageID $i$.
- All indices and coefficients are as retrieved and in original file order.