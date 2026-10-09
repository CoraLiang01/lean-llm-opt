Let $x_{ij}$ be the number of units of product $j$ (ProductName from products.csv) to be placed on shelf $i$ (ShelfID from capacity.csv). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Let $S$ be the set of shelves (ShelfID):  
  $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$

- Let $P$ be the set of products (ProductName):  
  $P = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$

- For each shelf $i \in S$, let $C_i$ be its capacity (Capacity from capacity.csv):

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

- For each product $j \in P$, let $v_j$ be its value (Value from products.csv), and $w_j$ its weight (Weight from products.csv):

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

**Data Used (in source order):**

- Shelves (ShelfID, Capacity):

  1. ShelfID: 1, Capacity: 5.0
  2. ShelfID: 2, Capacity: 7.0
  3. ShelfID: 3, Capacity: 6.0
  4. ShelfID: 4, Capacity: 8.0
  5. ShelfID: 5, Capacity: 5.5
  6. ShelfID: 6, Capacity: 9.0
  7. ShelfID: 7, Capacity: 6.5
  8. ShelfID: 8, Capacity: 7.5
  9. ShelfID: 9, Capacity: 8.2
  10. ShelfID: 10, Capacity: 5.7

- Products (ProductName, Value, Weight):

  1. Smartphone, 200, 1.0
  2. Laptop, 1500, 5.0
  3. Headphones, 100, 0.5
  4. Camera, 800, 2.0
  5. Smartwatch, 250, 0.3
  6. Tablet, 600, 1.5
  7. Bluetooth Speaker, 150, 1.0
  8. Keyboard, 80, 0.8
  9. Mouse, 50, 0.2
  10. Monitor, 300, 3.0
  11. Printer, 400, 4.0
  12. External Hard Drive, 120, 0.5
  13. Router, 60, 0.3
  14. Power Bank, 40, 0.4
  15. Memory Card, 30, 0.05
  16. USB Flash Drive, 25, 0.02
  17. Smart Home Hub, 100, 0.6
  18. Gaming Console, 500, 4.0
  19. Fitness Tracker, 90, 0.2
  20. E-Reader, 180, 0.5

---

**Summary:**

\[
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10, \; j=1,\ldots,20
\end{align*}
\]

where $v_j$ and $w_j$ are as listed above, and $C_i$ as per each shelf.