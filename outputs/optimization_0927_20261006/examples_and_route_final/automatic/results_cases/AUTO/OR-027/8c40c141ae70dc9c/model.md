Let $x_i$ denote the number of units of Organ product $i$ to be fulfilled, for each $i$ in the set of Organ products below.

Parameters (from the data):

- Organ products (in source order):
    1. Organic Fruits
    2. Organic Staples
    3. Organic Vegetables

- Revenues:
    - Organic Fruits: $60.8$
    - Organic Staples: $918.45$
    - Organic Vegetables: $77.52$

- Initial Inventory:
    - Organic Fruits: $5,\!034,\!020.0$
    - Organic Staples: $5,\!589,\!290.0$
    - Organic Vegetables: $5,\!202,\!710.0$

- Demand:
    - Organic Fruits: $678,\!906$
    - Organic Staples: $749,\!927$
    - Organic Vegetables: $699,\!808$

Model:

Objective:
\[
\max \; 60.8\, x_1 + 918.45\, x_2 + 77.52\, x_3
\]

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{5,\!034,\!020.0,\; 678,\!906\} \\
& 0 \leq x_2 \leq \min\{5,\!589,\!290.0,\; 749,\!927\} \\
& 0 \leq x_3 \leq \min\{5,\!202,\!710.0,\; 699,\!808\} \\
& x_1,\, x_2,\, x_3 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Where:
- $x_1$ = units of Organic Fruits fulfilled
- $x_2$ = units of Organic Staples fulfilled
- $x_3$ = units of Organic Vegetables fulfilled

All variables are nonnegative integers, and for each product, the number fulfilled cannot exceed either the initial inventory or the demand.