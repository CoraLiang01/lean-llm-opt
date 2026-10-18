##### Decision Variables

$x_i \geq 0$ (integer): Number of units of Organ product $i$ to fulfill, for each $i \in \{\text{Organic Fruits}, \text{Organic Staples}, \text{Organic Vegetables}\}$.

##### Parameters

- $\text{Revenue}_i$: Revenue per unit of product $i$.
- $\text{InitialInventory}_i$: Initial inventory available for product $i$.
- $\text{Demand}_i$: Demand for product $i$.

Values:

| $i$                  | $\text{Revenue}_i$ | $\text{InitialInventory}_i$ | $\text{Demand}_i$ |
|----------------------|-------------------|-----------------------------|-------------------|
| Organic Fruits       | 60.8              | 5,034,020                   | 678,906           |
| Organic Staples      | 918.45            | 5,589,290                   | 749,927           |
| Organic Vegetables   | 77.52             | 5,202,710                   | 699,808           |

##### Objective Function

\[
\max \sum_{i} \text{Revenue}_i \cdot x_i
\]

##### Constraints

1. Inventory and demand limits:
   \[
   0 \leq x_i \leq \min(\text{InitialInventory}_i,\, \text{Demand}_i), \quad \forall i
   \]

   Explicitly:
   - $0 \leq x_{\text{Organic Fruits}} \leq 678,906$
   - $0 \leq x_{\text{Organic Staples}} \leq 749,927$
   - $0 \leq x_{\text{Organic Vegetables}} \leq 699,808$

2. $x_i$ are integer variables.

##### Complete Model

\[
\begin{align*}
\max\quad & 60.8\, x_{\text{Organic Fruits}} + 918.45\, x_{\text{Organic Staples}} + 77.52\, x_{\text{Organic Vegetables}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{Organic Fruits}} \leq 678,906 \\
& 0 \leq x_{\text{Organic Staples}} \leq 749,927 \\
& 0 \leq x_{\text{Organic Vegetables}} \leq 699,808 \\
& x_{\text{Organic Fruits}},\, x_{\text{Organic Staples}},\, x_{\text{Organic Vegetables}} \in \mathbb{Z}_{\geq 0}
\end{align*}
\]