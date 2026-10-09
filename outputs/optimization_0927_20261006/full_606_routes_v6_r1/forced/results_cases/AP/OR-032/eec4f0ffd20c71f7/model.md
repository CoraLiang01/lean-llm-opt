##### Decision Variables:

For each product $i$ in the set of ‘Books’ products $\mathcal{B}$, let $x_i$ denote the number of units of product $i$ to be fulfilled.

##### Sets and Parameters:

Let $\mathcal{B} = \{\text{Books\_15.15}, \text{Books\_30.3}, \text{Books\_45.45}, \text{Books\_60.6}, \text{Books\_75.75}\}$

For each $i \in \mathcal{B}$:

- $r_i$: Revenue per unit of product $i$
- $s_i$: Initial Inventory of product $i$
- $d_i$: Demand for product $i$

Parameter values:

\[
\begin{array}{l|c|c|c}
\text{Product} & r_i & s_i & d_i \\
\hline
\text{Books\_15.15} & 15.15 & 9920.0 & 1980 \\
\text{Books\_30.3} & 30.3 & 20160.0 & 3024 \\
\text{Books\_45.45} & 45.45 & 30000.0 & 4536 \\
\text{Books\_60.6} & 60.6 & 38360.0 & 5601 \\
\text{Books\_75.75} & 75.75 & 51450.0 & 7567 \\
\end{array}
\]

##### Objective Function:

\[
\max \sum_{i \in \mathcal{B}} r_i x_i
\]

##### Constraints:

For each $i \in \mathcal{B}$:

\[
0 \leq x_i \leq \min(s_i, d_i)
\]

##### Variable Domains:

\[
x_i \geq 0 \quad \text{and integer, for all } i \in \mathcal{B}
\]

##### Retrieved Information

{
  "Books_15.15": {
    "Revenue": 15.15,
    "Initial Inventory": 9920.0,
    "Demand": 1980
  },
  "Books_30.3": {
    "Revenue": 30.3,
    "Initial Inventory": 20160.0,
    "Demand": 3024
  },
  "Books_45.45": {
    "Revenue": 45.45,
    "Initial Inventory": 30000.0,
    "Demand": 4536
  },
  "Books_60.6": {
    "Revenue": 60.6,
    "Initial Inventory": 38360.0,
    "Demand": 5601
  },
  "Books_75.75": {
    "Revenue": 75.75,
    "Initial Inventory": 51450.0,
    "Demand": 7567
  }
}

##### Summary

- Decision variables $x_i$ represent the fulfilled units for each ‘Books’ product $i$.
- Objective: maximize total revenue from all fulfilled ‘Books’ products.
- Constraints: For each product, fulfilled units cannot exceed either initial inventory or demand, and must be non-negative integers.