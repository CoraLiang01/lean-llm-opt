Let $x_i$ be the number of units of product $i$ to fulfill, for each product $i$ classified as ‘ELE-S’ (with Product_Reference as below). All variables are nonnegative integers.

**Parameters:**

- $r_i$ = Revenue per unit of product $i$ (from ‘Revenue’ column)
- $d_i$ = Demand for product $i$ (from ‘Demand’ column)
- $s_i$ = Initial Inventory for product $i$ (from ‘Initial Inventory’ column)

**Products and Data:**

| Product_Reference         | $r_i$ | $d_i$ | $s_i$   |
|--------------------------|-------|-------|---------|
| ELE-SMA-10000463         | 4.0   | 295   | 2000.0  |
| ELE-SMA-10000487         | 14.0  | 1002  | 7000.0  |
| ELE-SMA-10003333         | 14.0  | 958   | 7000.0  |
| ELE-SMA-10009012         | 4.0   | 777   | 6000.0  |
| ELE-SMA-10009999         | 4.0   | 271   | 2000.0  |
| ELE-SMA-10011234         | 4.0   | 244   | 2000.0  |
| ELE-SMA-10027456         | 14.0  | 990   | 7000.0  |
| ELE-SMA-10028567         | 14.0  | 1000  | 7000.0  |

---

**Mathematical Model**

**Objective:**
\[
\max \sum_{i} r_i x_i
\]

**Subject to:**

\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \\
& x_i \leq s_i, \quad \forall i \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i
\end{align*}
\]

Where $i$ indexes the following Product_Reference values:

- ELE-SMA-10000463
- ELE-SMA-10000487
- ELE-SMA-10003333
- ELE-SMA-10009012
- ELE-SMA-10009999
- ELE-SMA-10011234
- ELE-SMA-10027456
- ELE-SMA-10028567

**Decision variables:**  
$x_i$ = number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

**Parameters:**  
$r_i$ = Revenue per unit (see table above)  
$d_i$ = Demand (see table above)  
$s_i$ = Initial Inventory (see table above)

**Constraints:**  
For each product $i$:
- Cannot fulfill more than demand: $x_i \leq d_i$
- Cannot fulfill more than available inventory: $x_i \leq s_i$
- Nonnegative integer: $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**  
Maximize total revenue from fulfilled units.