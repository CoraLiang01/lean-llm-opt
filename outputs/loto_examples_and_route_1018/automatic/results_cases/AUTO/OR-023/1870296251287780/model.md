Let $x_i$ denote the number of units of product $i$ (with Product_Reference as below) to fulfill.

**Parameters:**

- For each product $i$:
    - $r_i$: Revenue per unit (from ‘Revenue’ column)
    - $d_i$: Demand (from ‘Demand’ column)
    - $s_i$: Initial Inventory (from ‘Initial Inventory’ column)

**Sets:**

- $i \in \{$
    ELE-SMA-10000463,
    ELE-SMA-10000487,
    ELE-SMA-10003333,
    ELE-SMA-10009012,
    ELE-SMA-10009999,
    ELE-SMA-10011234,
    ELE-SMA-10027456,
    ELE-SMA-10028567
$\}$

**Model:**

Maximize total revenue:
$$
\max \quad 4.0\,x_{\text{ELE-SMA-10000463}}
+ 14.0\,x_{\text{ELE-SMA-10000487}}
+ 14.0\,x_{\text{ELE-SMA-10003333}}
+ 4.0\,x_{\text{ELE-SMA-10009012}}
+ 4.0\,x_{\text{ELE-SMA-10009999}}
+ 4.0\,x_{\text{ELE-SMA-10011234}}
+ 14.0\,x_{\text{ELE-SMA-10027456}}
+ 14.0\,x_{\text{ELE-SMA-10028567}}
$$

Subject to, for each product $i$:

- Demand fulfillment:
  $$
  0 \leq x_i \leq d_i
  $$
- Inventory limit:
  $$
  x_i \leq s_i
  $$

Explicitly, for each product:

1. $0 \leq x_{\text{ELE-SMA-10000463}} \leq \min\{295,\ 2000.0\}$
2. $0 \leq x_{\text{ELE-SMA-10000487}} \leq \min\{1002,\ 7000.0\}$
3. $0 \leq x_{\text{ELE-SMA-10003333}} \leq \min\{958,\ 7000.0\}$
4. $0 \leq x_{\text{ELE-SMA-10009012}} \leq \min\{777,\ 6000.0\}$
5. $0 \leq x_{\text{ELE-SMA-10009999}} \leq \min\{271,\ 2000.0\}$
6. $0 \leq x_{\text{ELE-SMA-10011234}} \leq \min\{244,\ 2000.0\}$
7. $0 \leq x_{\text{ELE-SMA-10027456}} \leq \min\{990,\ 7000.0\}$
8. $0 \leq x_{\text{ELE-SMA-10028567}} \leq \min\{1000,\ 7000.0\}$

**Variable domains:**

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

**Where:**

\[
\begin{array}{llll}
\text{Product\_Reference} & r_i & d_i & s_i \\
\hline
\text{ELE-SMA-10000463} & 4.0 & 295 & 2000.0 \\
\text{ELE-SMA-10000487} & 14.0 & 1002 & 7000.0 \\
\text{ELE-SMA-10003333} & 14.0 & 958 & 7000.0 \\
\text{ELE-SMA-10009012} & 4.0 & 777 & 6000.0 \\
\text{ELE-SMA-10009999} & 4.0 & 271 & 2000.0 \\
\text{ELE-SMA-10011234} & 4.0 & 244 & 2000.0 \\
\text{ELE-SMA-10027456} & 14.0 & 990 & 7000.0 \\
\text{ELE-SMA-10028567} & 14.0 & 1000 & 7000.0 \\
\end{array}
\]

**Summary:**

Maximize total revenue from fulfilling demand for each ‘ELE-S’ product, subject to not exceeding both demand and initial inventory for each product, with integer, nonnegative fulfillment quantities.