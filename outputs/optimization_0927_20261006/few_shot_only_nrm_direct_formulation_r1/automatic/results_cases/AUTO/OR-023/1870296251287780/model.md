**Index Set:**  
Let $I$ be the set of all products with $Product\_Reference$ starting with ‘ELE-S’ (i.e., all ‘ELE-S’ products), indexed by $i$.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from ‘Revenue’ column)
- $d_i$: Demand for product $i$ (from ‘Demand’ column)
- $I_i$: Initial inventory for product $i$ (from ‘Initial Inventory’ column)

**Variables:**  
For each $i \in I$:
- $x_i$: Number of units of product $i$ to fulfill, integer, $0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**  
For each $i \in I$:
1. Inventory constraint: $x_i \leq I_i$
2. Demand constraint: $x_i \leq d_i$
3. Non-negativity and integrality: $x_i \in \mathbb{Z}_{\geq 0}$

---

**Retrieved Information (ELE-S products, in source order):**

| $i$ | Product_Reference         | $A_i$ (Revenue) | $d_i$ (Demand) | $I_i$ (Initial Inventory) |
|-----|--------------------------|-----------------|---------------|--------------------------|
| 1   | ELE-SMA-10000463         | 4.0             | 295           | 2000.0                   |
| 2   | ELE-SMA-10000487         | 14.0            | 1002          | 7000.0                   |
| 3   | ELE-SMA-10003333         | 14.0            | 958           | 7000.0                   |
| 4   | ELE-SMA-10009012         | 4.0             | 777           | 6000.0                   |
| 5   | ELE-SMA-10009999         | 4.0             | 271           | 2000.0                   |
| 6   | ELE-SMA-10011234         | 4.0             | 244           | 2000.0                   |
| 7   | ELE-SMA-10027456         | 14.0            | 990           | 7000.0                   |
| 8   | ELE-SMA-10028567         | 14.0            | 1000          | 7000.0                   |

**Explicit Model:**

**Sets:**  
$I = \{$  
$\quad$1: ELE-SMA-10000463,  
$\quad$2: ELE-SMA-10000487,  
$\quad$3: ELE-SMA-10003333,  
$\quad$4: ELE-SMA-10009012,  
$\quad$5: ELE-SMA-10009999,  
$\quad$6: ELE-SMA-10011234,  
$\quad$7: ELE-SMA-10027456,  
$\quad$8: ELE-SMA-10028567  
$\}$

**Parameters:**  
\[
\begin{align*}
A &= [4.0,\, 14.0,\, 14.0,\, 4.0,\, 4.0,\, 4.0,\, 14.0,\, 14.0] \\
d &= [295,\, 1002,\, 958,\, 777,\, 271,\, 244,\, 990,\, 1000] \\
I &= [2000.0,\, 7000.0,\, 7000.0,\, 6000.0,\, 2000.0,\, 2000.0,\, 7000.0,\, 7000.0] \\
\end{align*}
\]

**Variables:**  
For $i = 1, \ldots, 8$:
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

**Objective:**  
\[
\max \left( 4.0\,x_1 + 14.0\,x_2 + 14.0\,x_3 + 4.0\,x_4 + 4.0\,x_5 + 4.0\,x_6 + 14.0\,x_7 + 14.0\,x_8 \right)
\]

**Constraints:**  
For $i = 1, \ldots, 8$:
\[
\begin{align*}
x_i &\leq d_i \\
x_i &\leq I_i \\
x_i &\geq 0,\quad x_i \in \mathbb{Z}
\end{align*}
\]

**Full Data Table:**

| $i$ | Product_Reference         | $A_i$ | $d_i$ | $I_i$  |
|-----|--------------------------|-------|-------|--------|
| 1   | ELE-SMA-10000463         | 4.0   | 295   | 2000.0 |
| 2   | ELE-SMA-10000487         | 14.0  | 1002  | 7000.0 |
| 3   | ELE-SMA-10003333         | 14.0  | 958   | 7000.0 |
| 4   | ELE-SMA-10009012         | 4.0   | 777   | 6000.0 |
| 5   | ELE-SMA-10009999         | 4.0   | 271   | 2000.0 |
| 6   | ELE-SMA-10011234         | 4.0   | 244   | 2000.0 |
| 7   | ELE-SMA-10027456         | 14.0  | 990   | 7000.0 |
| 8   | ELE-SMA-10028567         | 14.0  | 1000  | 7000.0 |

**Summary:**  
Maximize
\[
4.0\,x_1 + 14.0\,x_2 + 14.0\,x_3 + 4.0\,x_4 + 4.0\,x_5 + 4.0\,x_6 + 14.0\,x_7 + 14.0\,x_8
\]
subject to, for $i = 1, \ldots, 8$,
\[
0 \leq x_i \leq \min\{d_i, I_i\},\quad x_i \in \mathbb{Z}
\]
with all parameters as listed above.