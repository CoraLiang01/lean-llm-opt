Let $x_i$ be the number of units of product $i$ (where $i$ indexes the following Books products) to fulfill.

**Sets and Parameters:**

- Products (Books):
    - Books_15.15: Revenue = 15.15, Demand = 1980, Initial Inventory = 9920.0
    - Books_30.3: Revenue = 30.3, Demand = 3024, Initial Inventory = 20160.0
    - Books_45.45: Revenue = 45.45, Demand = 4536, Initial Inventory = 30000.0
    - Books_60.6: Revenue = 60.6, Demand = 5601, Initial Inventory = 38360.0
    - Books_75.75: Revenue = 75.75, Demand = 7567, Initial Inventory = 51450.0

**Decision Variables:**

- $x_i \in \mathbb{Z}_{\geq 0}$, for each Books product $i$

**Objective:**

$$
\max\ 15.15\, x_{\text{Books\_15.15}} + 30.3\, x_{\text{Books\_30.3}} + 45.45\, x_{\text{Books\_45.45}} + 60.6\, x_{\text{Books\_60.6}} + 75.75\, x_{\text{Books\_75.75}}
$$

**Constraints:**

For each product $i$:

- $0 \leq x_i \leq \min\{\text{Demand}_i,\, \text{Initial Inventory}_i\}$

Explicitly:

\[
\begin{align*}
0 &\leq x_{\text{Books\_15.15}} \leq 1980 \\
0 &\leq x_{\text{Books\_30.3}} \leq 3024 \\
0 &\leq x_{\text{Books\_45.45}} \leq 4536 \\
0 &\leq x_{\text{Books\_60.6}} \leq 5601 \\
0 &\leq x_{\text{Books\_75.75}} \leq 7567 \\
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
\]

**Where:**

- $x_{\text{Books\_15.15}}$ = units of Books_15.15 fulfilled
- $x_{\text{Books\_30.3}}$ = units of Books_30.3 fulfilled
- $x_{\text{Books\_45.45}}$ = units of Books_45.45 fulfilled
- $x_{\text{Books\_60.6}}$ = units of Books_60.6 fulfilled
- $x_{\text{Books\_75.75}}$ = units of Books_75.75 fulfilled