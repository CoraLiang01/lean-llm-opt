Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $i$ indexes ShelfID from capacity.csv and $j$ indexes ProductName from products.csv. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves (ShelfID):  
  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$

- Let $P$ be the set of products (ProductName):  
  $P = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$

- For each shelf $i \in S$, let $C_i$ be its capacity:
  \begin{align*}
  C_1 &= 5 \\
  C_2 &= 7 \\
  C_3 &= 6 \\
  C_4 &= 8 \\
  C_5 &= 5.5 \\
  C_6 &= 9 \\
  C_7 &= 6.5 \\
  C_8 &= 7.5 \\
  C_9 &= 8.2 \\
  C_{10} &= 5.7 \\
  \end{align*}

- For each product $j \in P$, let $v_j$ be its value and $w_j$ its weight:
  \begin{align*}
  &\text{Smartphone:} \quad v = 200, \quad w = 1 \\
  &\text{Laptop:} \quad v = 1500, \quad w = 5 \\
  &\text{Headphones:} \quad v = 100, \quad w = 0.5 \\
  &\text{Camera:} \quad v = 800, \quad w = 2 \\
  &\text{Smartwatch:} \quad v = 250, \quad w = 0.3 \\
  &\text{Tablet:} \quad v = 600, \quad w = 1.5 \\
  &\text{Bluetooth Speaker:} \quad v = 150, \quad w = 1 \\
  &\text{Keyboard:} \quad v = 80, \quad w = 0.8 \\
  &\text{Mouse:} \quad v = 50, \quad w = 0.2 \\
  &\text{Monitor:} \quad v = 300, \quad w = 3 \\
  &\text{Printer:} \quad v = 400, \quad w = 4 \\
  &\text{External Hard Drive:} \quad v = 120, \quad w = 0.5 \\
  &\text{Router:} \quad v = 60, \quad w = 0.3 \\
  &\text{Power Bank:} \quad v = 40, \quad w = 0.4 \\
  &\text{Memory Card:} \quad v = 30, \quad w = 0.05 \\
  &\text{USB Flash Drive:} \quad v = 25, \quad w = 0.02 \\
  &\text{Smart Home Hub:} \quad v = 100, \quad w = 0.6 \\
  &\text{Gaming Console:} \quad v = 500, \quad w = 4 \\
  &\text{Fitness Tracker:} \quad v = 90, \quad w = 0.2 \\
  &\text{E-Reader:} \quad v = 180, \quad w = 0.5 \\
  \end{align*}

---

**Mathematical Model:**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
$$

**Objective:**
$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

1. **Shelf Capacity Constraints:**  
   For each shelf $i \in S$,
   $$
   \sum_{j \in P} w_j \cdot x_{ij} \leq C_i
   $$

2. **Minimum Placement of First Product (Smartphone):**
   $$
   \sum_{i \in S} x_{i,\text{Smartphone}} \geq 5
   $$

3. **Nonnegativity and Integrality:**
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
   $$

---

**Parameter Table (source order):**

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