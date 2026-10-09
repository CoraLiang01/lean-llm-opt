Let $x_{ij}$ be the number of units of air conditioner type $j$ (ProductName) to be placed in storage area $i$ (StorageID). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Storage areas (from capacity.csv, in source order):

  | StorageID |
  |-----------|
  | 1         |
  | 2         |
  | 3         |
  | 4         |
  | 5         |
  | 6         |
  | 7         |
  | 8         |
  | 9         |
  | 10        |
  | 11        |
  | 12        |
  | 13        |
  | 14        |
  | 15        |

  Capacities:

  $C_1 = 1083$, $C_2 = 1840$, $C_3 = 770$, $C_4 = 1299$, $C_5 = 1259$, $C_6 = 543$, $C_7 = 1831$, $C_8 = 855$, $C_9 = 619$, $C_{10} = 637$, $C_{11} = 935$, $C_{12} = 626$, $C_{13} = 1457$, $C_{14} = 1198$, $C_{15} = 837$

- Air conditioner types (from products.csv, in source order):

  | ProductName           | Value | Weight |
  |----------------------|-------|--------|
  | Window Unit          | 4811  | 114    |
  | Portable Unit        | 1130  | 200    |
  | Split System         | 1611  | 106    |
  | Ductless System      | 3368  | 256    |
  | Central AC           | 2135  | 268    |
  | Hybrid AC            | 1046  | 185    |
  | Geothermal AC        | 4030  | 299    |
  | Smart AC             | 3761  | 131    |
  | Evaporative Cooler   | 3523  | 139    |
  | Package Unit         | 1701  | 105    |

Let $V_j$ be the Value and $W_j$ the Weight for product $j$ as above.

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of air conditioner type } j \text{ placed in storage area } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,15\}} \sum_{j \in \{\text{Window Unit}, \text{Portable Unit}, \ldots, \text{Package Unit}\}} V_j \cdot x_{ij}
$$

**Subject to:**

For each storage area $i$ (with StorageID as above):

$$
\sum_{j} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,15\}
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Explicitly, for each storage area $i$ (in source order):**

- For StorageID 1: $114\,x_{1,\,\text{Window Unit}} + 200\,x_{1,\,\text{Portable Unit}} + 106\,x_{1,\,\text{Split System}} + 256\,x_{1,\,\text{Ductless System}} + 268\,x_{1,\,\text{Central AC}} + 185\,x_{1,\,\text{Hybrid AC}} + 299\,x_{1,\,\text{Geothermal AC}} + 131\,x_{1,\,\text{Smart AC}} + 139\,x_{1,\,\text{Evaporative Cooler}} + 105\,x_{1,\,\text{Package Unit}} \leq 1083$

- For StorageID 2: (same structure, $C_2 = 1840$)
- ...
- For StorageID 15: (same structure, $C_{15} = 837$)

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,15\},\ j \in \{\text{Window Unit}, \text{Portable Unit}, \text{Split System}, \text{Ductless System}, \text{Central AC}, \text{Hybrid AC}, \text{Geothermal AC}, \text{Smart AC}, \text{Evaporative Cooler}, \text{Package Unit}\}
$$

---

**All identifiers, coefficients, and constraints are as retrieved and in source order.**