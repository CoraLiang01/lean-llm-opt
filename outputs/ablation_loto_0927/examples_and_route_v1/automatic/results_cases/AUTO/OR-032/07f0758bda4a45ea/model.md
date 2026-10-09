Let $x_i$ be the number of units of product $i$ (where $i$ indexes the following five products) to be fulfilled.

Products and parameters:

| $i$ | Product Name         | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|-----|---------------------|-----------------|----------------|---------------------------|
| 1   | Books_15.15         | 15.15           | 1980           | 9920.0                    |
| 2   | Books_30.3          | 30.3            | 3024           | 20160.0                   |
| 3   | Books_45.45         | 45.45           | 4536           | 30000.0                   |
| 4   | Books_60.6          | 60.6            | 5601           | 38360.0                   |
| 5   | Books_75.75         | 75.75           | 7567           | 51450.0                   |

Mathematical Model:

Objective:
\[
\max \; 15.15\,x_1 + 30.3\,x_2 + 45.45\,x_3 + 60.6\,x_4 + 75.75\,x_5
\]

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{1980,\; 9920.0\} \\
& 0 \leq x_2 \leq \min\{3024,\; 20160.0\} \\
& 0 \leq x_3 \leq \min\{4536,\; 30000.0\} \\
& 0 \leq x_4 \leq \min\{5601,\; 38360.0\} \\
& 0 \leq x_5 \leq \min\{7567,\; 51450.0\} \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,5
\end{align*}
\]

Or, equivalently (since inventory always exceeds demand in this data):

\[
\begin{align*}
& 0 \leq x_1 \leq 1980 \\
& 0 \leq x_2 \leq 3024 \\
& 0 \leq x_3 \leq 4536 \\
& 0 \leq x_4 \leq 5601 \\
& 0 \leq x_5 \leq 7567 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,5
\end{align*}
\]

Where:
- $x_i$ = units of product $i$ fulfilled (decision variable, integer, nonnegative)
- $r_i$ = revenue per unit of product $i$
- $d_i$ = demand for product $i$
- $s_i$ = initial inventory for product $i$