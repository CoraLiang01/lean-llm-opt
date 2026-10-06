Let $x_i$ denote the number of units of Organ product $i$ to be fulfilled, for each $i$ in the set of Organ products:

- $i=1$: Organic Fruits
- $i=2$: Organic Staples
- $i=3$: Organic Vegetables

Parameters:
- Revenue per unit: $r_i$
- Demand: $d_i$
- Initial Inventory: $s_i$

Given data (in source order):

| Organ               | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|---------------------|-----------------|----------------|---------------------------|
| Organic Fruits      | 60.8            | 678906         | 5034020.0                 |
| Organic Staples     | 918.45          | 749927         | 5589290.0                 |
| Organic Vegetables  | 77.52           | 699808         | 5202710.0                 |

Objective:
\[
\max \; 60.8\,x_1 + 918.45\,x_2 + 77.52\,x_3
\]

Subject to:
\[
\begin{align*}
& 0 \leq x_1 \leq \min\{5034020.0,\;678906\} \\
& 0 \leq x_2 \leq \min\{5589290.0,\;749927\} \\
& 0 \leq x_3 \leq \min\{5202710.0,\;699808\} \\
& x_1,\,x_2,\,x_3 \in \mathbb{Z}_{\geq 0}
\end{align*}
\]

Or, equivalently, for each $i$:
\[
0 \leq x_i \leq \min\{\text{Initial Inventory}_i,\,\text{Demand}_i\}, \quad x_i \in \mathbb{Z}_{\geq 0}
\]

Where:
- $x_1$: units of Organic Fruits fulfilled
- $x_2$: units of Organic Staples fulfilled
- $x_3$: units of Organic Vegetables fulfilled