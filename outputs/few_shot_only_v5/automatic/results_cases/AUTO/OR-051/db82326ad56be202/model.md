Let $x_{ij}$ be the number of units of coffee product $j$ to be placed in cabinet $i$, where $i \in \{1,2,\ldots,10\}$ indexes cabinets (with CabinetID as below) and $j$ indexes the 18 coffee products (with ProductName as below). All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Cabinets (indexed by $i$ with CabinetID):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10

- Cabinet capacities $C_i$:
  - $C_1 = 400$
  - $C_2 = 600$
  - $C_3 = 500$
  - $C_4 = 700$
  - $C_5 = 450$
  - $C_6 = 650$
  - $C_7 = 550$
  - $C_8 = 750$
  - $C_9 = 480$
  - $C_{10} = 520$

- Coffee products (indexed by $j$ with ProductName, Value $v_j$, and Weight $w_j$):

| $j$ | ProductName         | $v_j$ | $w_j$ |
|-----|---------------------|-------|-------|
| 1   | Espresso Beans      | 100   | 1.0   |
| 2   | Colombian Roast     | 150   | 1.5   |
| 3   | Arabica Blend       | 80    | 1.2   |
| 4   | French Roast        | 120   | 1.3   |
| 5   | Italian Roast       | 130   | 1.4   |
| 6   | House Blend         | 110   | 1.1   |
| 7   | Sumatra Coffee     | 160   | 1.8   |
| 8   | Mocha Java          | 90    | 1.2   |
| 9   | Hazelnut Flavor     | 95    | 1.0   |
| 10  | Caramel Blend       | 105   | 1.3   |
| 11  | Vanilla Flavor      | 85    | 1.2   |
| 12  | Cappuccino Mix      | 140   | 1.5   |
| 13  | Pumpkin Spice       | 75    | 1.1   |
| 14  | Decaf Roast         | 60    | 1.0   |
| 15  | Organic Roast       | 170   | 1.6   |
| 16  | Cold Brew           | 115   | 1.4   |
| 17  | Peruvian Blend      | 155   | 1.7   |
| 18  | Kenyan AA           | 125   | 1.3   |

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{18} v_j \cdot x_{ij}
\]

**Subject to:**

- **Cabinet capacity constraints:**  
  For each cabinet $i$:
  \[
  \sum_{j=1}^{18} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
  \]

- **Nonnegativity and integrality:**
  \[
  x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\ j \in \{1,2,\ldots,18\}
  \]

---

**Where:**

- $x_{ij}$ = number of units of coffee product $j$ placed in cabinet $i$
- $v_j$ = value of product $j$ (see table above)
- $w_j$ = weight of product $j$ (see table above)
- $C_i$ = capacity of cabinet $i$ (see list above)

All identifiers and coefficients are as retrieved from the data.