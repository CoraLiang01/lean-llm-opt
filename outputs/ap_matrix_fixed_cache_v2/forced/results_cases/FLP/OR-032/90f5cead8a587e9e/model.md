##### Sets and Indices

Let $I$ be the set of all products classified as ‘Books’:
$$
I = \{\text{Books\_15.15},\ \text{Books\_30.3},\ \text{Books\_45.45},\ \text{Books\_60.6},\ \text{Books\_75.75}\}
$$

##### Parameters

For each $i \in I$:
- $r_i$: revenue per unit of product $i$
- $d_i$: demand for product $i$
- $s_i$: initial inventory of product $i$

From the data:
\[
\begin{array}{l|c|c|c}
\text{Product} & r_i\ (\text{Revenue}) & d_i\ (\text{Demand}) & s_i\ (\text{Initial Inventory}) \\
\hline
\text{Books\_15.15} & 15.15 & 1980 & 9920.0 \\
\text{Books\_30.3} & 30.3 & 3024 & 20160.0 \\
\text{Books\_45.45} & 45.45 & 4536 & 30000.0 \\
\text{Books\_60.6} & 60.6 & 5601 & 38360.0 \\
\text{Books\_75.75} & 75.75 & 7567 & 51450.0 \\
\end{array}
\]

##### Decision Variables

For each $i \in I$:
- $x_i \geq 0$: number of units of product $i$ to fulfill (continuous or integer, as appropriate).

##### Objective Function

\[
\max \sum_{i \in I} r_i x_i
\]

##### Constraints

1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min\{d_i,\, s_i\}, \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\max\quad & 15.15\, x_{\text{Books\_15.15}} + 30.3\, x_{\text{Books\_30.3}} + 45.45\, x_{\text{Books\_45.45}} + 60.6\, x_{\text{Books\_60.6}} + 75.75\, x_{\text{Books\_75.75}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{Books\_15.15}} \leq 1980 \\
& 0 \leq x_{\text{Books\_30.3}} \leq 3024 \\
& 0 \leq x_{\text{Books\_45.45}} \leq 4536 \\
& 0 \leq x_{\text{Books\_60.6}} \leq 5601 \\
& 0 \leq x_{\text{Books\_75.75}} \leq 7567 \\
& x_i \geq 0,\quad \forall i \in I
\end{align*}
\]

##### Retrieved Information

{
  "Books_15.15": {"Revenue": 15.15, "Demand": 1980, "Initial Inventory": 9920.0},
  "Books_30.3": {"Revenue": 30.3, "Demand": 3024, "Initial Inventory": 20160.0},
  "Books_45.45": {"Revenue": 45.45, "Demand": 4536, "Initial Inventory": 30000.0},
  "Books_60.6": {"Revenue": 60.6, "Demand": 5601, "Initial Inventory": 38360.0},
  "Books_75.75": {"Revenue": 75.75, "Demand": 7567, "Initial Inventory": 51450.0}
}