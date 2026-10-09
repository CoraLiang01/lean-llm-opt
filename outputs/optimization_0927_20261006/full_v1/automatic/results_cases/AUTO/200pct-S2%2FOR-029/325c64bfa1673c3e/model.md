Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$.

Let $I$ be the set of displays (shelves), indexed by $i$ with ShelfID as below.

Let $J$ be the set of products, indexed by $j$ with ProductName as below.

Let $v_j$ be the value of product $j$.

Let $w_j$ be the weight of product $j$.

Let $C_i$ be the capacity of display $i$.

The first product in source order is "Smartphone".

---

**Sets and Parameters (in source order):**

Displays (from capacity.csv):

| ShelfID | Capacity |
|---------|----------|
| 1       | 5        |
| 2       | 7        |
| 3       | 6        |
| 4       | 8        |
| 5       | 5.5      |
| 6       | 9        |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

Products (from products.csv):

| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Smartphone            | 200   | 1      |
| Laptop                | 1500  | 5      |
| Headphones            | 100   | 0.5    |
| Camera                | 800   | 2      |
| Smartwatch            | 250   | 0.3    |
| Tablet                | 600   | 1.5    |
| Bluetooth Speaker     | 150   | 1      |
| Keyboard              | 80    | 0.8    |
| Mouse                 | 50    | 0.2    |
| Monitor               | 300   | 3      |
| Printer               | 400   | 4      |
| External Hard Drive   | 120   | 0.5    |
| Router                | 60    | 0.3    |
| Power Bank            | 40    | 0.4    |
| Memory Card           | 30    | 0.05   |
| USB Flash Drive       | 25    | 0.02   |
| Smart Home Hub        | 100   | 0.6    |
| Gaming Console        | 500   | 4      |
| Fitness Tracker       | 90    | 0.2    |
| E-Reader              | 180   | 0.5    |

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{\text{Smartphone}, \text{Laptop}, \ldots, \text{E-Reader}\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

where $v_j$ is the value of product $j$ as listed above.

**Constraints:**

1. **Display Capacity Constraints (for each display $i$):**

$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the weight of product $j$ and $C_i$ is the capacity of display $i$ as listed above.

2. **Minimum Total Quantity of First Product ("Smartphone") Across All Displays:**

$$
\sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
$$

3. **Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

---

**Complete Numerical Formulation:**

Let $x_{ij}$ be the number of units of product $j$ placed on display $i$.

**Maximize:**
$$
\sum_{i=1}^{10} \Big[
200\, x_{i,\text{Smartphone}}
+ 1500\, x_{i,\text{Laptop}}
+ 100\, x_{i,\text{Headphones}}
+ 800\, x_{i,\text{Camera}}
+ 250\, x_{i,\text{Smartwatch}}
+ 600\, x_{i,\text{Tablet}}
+ 150\, x_{i,\text{Bluetooth Speaker}}
+ 80\, x_{i,\text{Keyboard}}
+ 50\, x_{i,\text{Mouse}}
+ 300\, x_{i,\text{Monitor}}
+ 400\, x_{i,\text{Printer}}
+ 120\, x_{i,\text{External Hard Drive}}
+ 60\, x_{i,\text{Router}}
+ 40\, x_{i,\text{Power Bank}}
+ 30\, x_{i,\text{Memory Card}}
+ 25\, x_{i,\text{USB Flash Drive}}
+ 100\, x_{i,\text{Smart Home Hub}}
+ 500\, x_{i,\text{Gaming Console}}
+ 90\, x_{i,\text{Fitness Tracker}}
+ 180\, x_{i,\text{E-Reader}}
\Big]
$$

**Subject to, for each display $i$:**

- For $i=1$ (ShelfID 1, $C_1=5$):

  $$
  1\, x_{1,\text{Smartphone}}
  + 5\, x_{1,\text{Laptop}}
  + 0.5\, x_{1,\text{Headphones}}
  + 2\, x_{1,\text{Camera}}
  + 0.3\, x_{1,\text{Smartwatch}}
  + 1.5\, x_{1,\text{Tablet}}
  + 1\, x_{1,\text{Bluetooth Speaker}}
  + 0.8\, x_{1,\text{Keyboard}}
  + 0.2\, x_{1,\text{Mouse}}
  + 3\, x_{1,\text{Monitor}}
  + 4\, x_{1,\text{Printer}}
  + 0.5\, x_{1,\text{External Hard Drive}}
  + 0.3\, x_{1,\text{Router}}
  + 0.4\, x_{1,\text{Power Bank}}
  + 0.05\, x_{1,\text{Memory Card}}
  + 0.02\, x_{1,\text{USB Flash Drive}}
  + 0.6\, x_{1,\text{Smart Home Hub}}
  + 4\, x_{1,\text{Gaming Console}}
  + 0.2\, x_{1,\text{Fitness Tracker}}
  + 0.5\, x_{1,\text{E-Reader}}
  \leq 5
  $$

- For $i=2$ (ShelfID 2, $C_2=7$):

  $$
  \ldots \leq 7
  $$

- For $i=3$ (ShelfID 3, $C_3=6$):

  $$
  \ldots \leq 6
  $$

- For $i=4$ (ShelfID 4, $C_4=8$):

  $$
  \ldots \leq 8
  $$

- For $i=5$ (ShelfID 5, $C_5=5.5$):

  $$
  \ldots \leq 5.5
  $$

- For $i=6$ (ShelfID 6, $C_6=9$):

  $$
  \ldots \leq 9
  $$

- For $i=7$ (ShelfID 7, $C_7=6.5$):

  $$
  \ldots \leq 6.5
  $$

- For $i=8$ (ShelfID 8, $C_8=7.5$):

  $$
  \ldots \leq 7.5
  $$

- For $i=9$ (ShelfID 9, $C_9=8.2$):

  $$
  \ldots \leq 8.2
  $$

- For $i=10$ (ShelfID 10, $C_{10}=5.7$):

  $$
  \ldots \leq 5.7
  $$

**Minimum total quantity of "Smartphone":**

$$
\sum_{i=1}^{10} x_{i,\text{Smartphone}} \geq 5
$$

**Nonnegativity and Integrality:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=\text{Smartphone},\ldots,\text{E-Reader}
$$

---

**All identifiers, coefficients, and constraints are as retrieved and in source order.**