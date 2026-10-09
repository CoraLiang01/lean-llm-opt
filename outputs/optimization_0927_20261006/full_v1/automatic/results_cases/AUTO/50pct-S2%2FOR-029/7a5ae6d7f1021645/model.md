Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, for $i \in \{1,2,\ldots,10\}$ and $j \in \{1,2,\ldots,20\}$, corresponding to the order of shelves and products as given below.

**Sets and Indices:**

- Shelves (Displays): $i \in \{1,2,\ldots,10\}$, with ShelfID as below.
- Products: $j \in \{1,2,\ldots,20\}$, with ProductName as below.

**Parameters:**

- $c_i$: Capacity of shelf $i$ (from "Capacity" column).
- $v_j$: Value of product $j$ (from "Value" column).
- $w_j$: Weight of product $j$ (from "Weight" column).

**Data (in source order):**

_Shelves (from capacity.csv):_

| $i$ | ShelfID | Capacity |
|----|---------|----------|
| 1  | 1       | 5        |
| 2  | 2       | 7        |
| 3  | 3       | 6        |
| 4  | 4       | 8        |
| 5  | 5       | 5.5      |
| 6  | 6       | 9        |
| 7  | 7       | 6.5      |
| 8  | 8       | 7.5      |
| 9  | 9       | 8.2      |
| 10 | 10      | 5.7      |

_Products (from products.csv, in source order):_

| $j$ | ProductName             | Value | Weight |
|-----|------------------------|-------|--------|
| 1   | Smartphone             | 200   | 1      |
| 2   | Laptop                 | 1500  | 5      |
| 3   | Headphones             | 100   | 0.5    |
| 4   | Camera                 | 800   | 2      |
| 5   | Smartwatch             | 250   | 0.3    |
| 6   | Tablet                 | 600   | 1.5    |
| 7   | Bluetooth Speaker      | 150   | 1      |
| 8   | Keyboard               | 80    | 0.8    |
| 9   | Mouse                  | 50    | 0.2    |
| 10  | Monitor                | 300   | 3      |
| 11  | Printer                | 400   | 4      |
| 12  | External Hard Drive    | 120   | 0.5    |
| 13  | Router                 | 60    | 0.3    |
| 14  | Power Bank             | 40    | 0.4    |
| 15  | Memory Card            | 30    | 0.05   |
| 16  | USB Flash Drive        | 25    | 0.02   |
| 17  | Smart Home Hub         | 100   | 0.6    |
| 18  | Gaming Console         | 500   | 4      |
| 19  | Fitness Tracker        | 90    | 0.2    |
| 20  | E-Reader               | 180   | 0.5    |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

**Subject to:**

1. **Shelf Capacity Constraints:**

For each shelf $i$ (with capacity $c_i$):

$$
\sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

2. **Minimum Placement of First Product (Smartphone):**

$$
\sum_{i=1}^{10} x_{i1} \geq 5
$$

3. **Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**Parameter Values (in source order):**

- Shelf capacities $c_i$:

  - $c_1 = 5$, $c_2 = 7$, $c_3 = 6$, $c_4 = 8$, $c_5 = 5.5$, $c_6 = 9$, $c_7 = 6.5$, $c_8 = 7.5$, $c_9 = 8.2$, $c_{10} = 5.7$

- Product values $v_j$ and weights $w_j$:

  - $v_1 = 200$, $w_1 = 1$ (Smartphone)
  - $v_2 = 1500$, $w_2 = 5$ (Laptop)
  - $v_3 = 100$, $w_3 = 0.5$ (Headphones)
  - $v_4 = 800$, $w_4 = 2$ (Camera)
  - $v_5 = 250$, $w_5 = 0.3$ (Smartwatch)
  - $v_6 = 600$, $w_6 = 1.5$ (Tablet)
  - $v_7 = 150$, $w_7 = 1$ (Bluetooth Speaker)
  - $v_8 = 80$, $w_8 = 0.8$ (Keyboard)
  - $v_9 = 50$, $w_9 = 0.2$ (Mouse)
  - $v_{10} = 300$, $w_{10} = 3$ (Monitor)
  - $v_{11} = 400$, $w_{11} = 4$ (Printer)
  - $v_{12} = 120$, $w_{12} = 0.5$ (External Hard Drive)
  - $v_{13} = 60$, $w_{13} = 0.3$ (Router)
  - $v_{14} = 40$, $w_{14} = 0.4$ (Power Bank)
  - $v_{15} = 30$, $w_{15} = 0.05$ (Memory Card)
  - $v_{16} = 25$, $w_{16} = 0.02$ (USB Flash Drive)
  - $v_{17} = 100$, $w_{17} = 0.6$ (Smart Home Hub)
  - $v_{18} = 500$, $w_{18} = 4$ (Gaming Console)
  - $v_{19} = 90$, $w_{19} = 0.2$ (Fitness Tracker)
  - $v_{20} = 180$, $w_{20} = 0.5$ (E-Reader)

---

**Summary:**

Maximize total value of products allocated to shelves, subject to shelf weight capacities, a minimum total allocation of the first product (Smartphone) across all shelves, and nonnegative integer variables for all allocations. All identifiers and coefficients are as retrieved and in source order.