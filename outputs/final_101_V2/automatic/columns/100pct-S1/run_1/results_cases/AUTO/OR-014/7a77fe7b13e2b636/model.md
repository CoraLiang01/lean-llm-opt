Let $x_{ij}$ be the number of units of product $j$ (ProductName) to be placed on shelf $i$ (ShelfID). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves, indexed by ShelfID from capacity.csv:

  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$

- Let $P$ be the set of products, indexed by ProductName from products.csv:

  $P = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$

- For each shelf $i \in S$, let $C_i$ be its capacity:

  \[
  \begin{align*}
  C_1 &= 5.0 \\
  C_2 &= 7.0 \\
  C_3 &= 6.0 \\
  C_4 &= 8.0 \\
  C_5 &= 5.5 \\
  C_6 &= 9.0 \\
  C_7 &= 6.5 \\
  C_8 &= 7.5 \\
  C_9 &= 8.2 \\
  C_{10} &= 5.7 \\
  \end{align*}
  \]

- For each product $j \in P$, let $v_j$ be its value and $w_j$ its weight:

  \[
  \begin{align*}
  &\text{Smartphone:} \quad v = 200, \quad w = 1.0 \\
  &\text{Laptop:} \quad v = 1500, \quad w = 5.0 \\
  &\text{Headphones:} \quad v = 100, \quad w = 0.5 \\
  &\text{Camera:} \quad v = 800, \quad w = 2.0 \\
  &\text{Smartwatch:} \quad v = 250, \quad w = 0.3 \\
  &\text{Tablet:} \quad v = 600, \quad w = 1.5 \\
  &\text{Bluetooth Speaker:} \quad v = 150, \quad w = 1.0 \\
  &\text{Keyboard:} \quad v = 80, \quad w = 0.8 \\
  &\text{Mouse:} \quad v = 50, \quad w = 0.2 \\
  &\text{Monitor:} \quad v = 300, \quad w = 3.0 \\
  &\text{Printer:} \quad v = 400, \quad w = 4.0 \\
  &\text{External Hard Drive:} \quad v = 120, \quad w = 0.5 \\
  &\text{Router:} \quad v = 60, \quad w = 0.3 \\
  &\text{Power Bank:} \quad v = 40, \quad w = 0.4 \\
  &\text{Memory Card:} \quad v = 30, \quad w = 0.05 \\
  &\text{USB Flash Drive:} \quad v = 25, \quad w = 0.02 \\
  &\text{Smart Home Hub:} \quad v = 100, \quad w = 0.6 \\
  &\text{Gaming Console:} \quad v = 500, \quad w = 4.0 \\
  &\text{Fitness Tracker:} \quad v = 90, \quad w = 0.2 \\
  &\text{E-Reader:} \quad v = 180, \quad w = 0.5 \\
  \end{align*}
  \]

---

### Mathematical Model

**Decision Variables:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \forall j \in P
\]

**Objective:**

\[
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Subject to:**

For each shelf $i \in S$:

\[
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
\]

For all $i \in S$, $j \in P$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Where:**

- $x_{ij}$ = number of units of product $j$ placed on shelf $i$
- $v_j$ = value of product $j$ (see above)
- $w_j$ = weight of product $j$ (see above)
- $C_i$ = capacity of shelf $i$ (see above)
- $S$ = $\{1,2,3,4,5,6,7,8,9,10\}$
- $P$ = set of 20 products listed above

All data and identifiers are used exactly as retrieved, in original order.