##### Decision Variables

$x_i \geq 0$: quantity fulfilled for each car model $i$ classified under ‘FDK57’.

##### Objective Function

$\max \sum_{i \in I} r_i x_i$

where $r_i$ is the revenue per unit for car model $i$.

##### Constraints

1. Inventory and demand limits: $0 \leq x_i \leq \min\{\text{Initial Inventory}_i, \text{Demand}_i\},\quad \forall i \in I$
2. $x_i$ is continuous and nonnegative.

##### Sets and Parameters

Let $I$ be the set of all car models with classification ‘FDK57’. For each $i \in I$:

- $r_i$: Revenue per unit for model $i$ (from ‘Revenue’ column)
- $\text{Initial Inventory}_i$: Initial inventory for model $i$
- $\text{Demand}_i$: Demand for model $i$

##### Data

The following are all car models with classification ‘FDK57’ and their parameters:

| Product Name | Revenue   | Demand | Initial Inventory |
|--------------|-----------|--------|------------------|
| FDK57        | 119.144   | 30     | 200              |
| FDK57        | 119.144   | 40     | 100              |
| FDK57        | 120.144   | 50     | 150              |
| FDK57        | 121.244   | 30     | 200              |
| FDK57        | 120.544   | 10     | 150              |

So, $I = \{\text{FDK57}_1, \text{FDK57}_2, \text{FDK57}_3, \text{FDK57}_4, \text{FDK57}_5\}$, with the following parameters:

- $\text{FDK57}_1$: $r_1 = 119.144$, $\text{Demand}_1 = 30$, $\text{Initial Inventory}_1 = 200$
- $\text{FDK57}_2$: $r_2 = 119.144$, $\text{Demand}_2 = 40$, $\text{Initial Inventory}_2 = 100$
- $\text{FDK57}_3$: $r_3 = 120.144$, $\text{Demand}_3 = 50$, $\text{Initial Inventory}_3 = 150$
- $\text{FDK57}_4$: $r_4 = 121.244$, $\text{Demand}_4 = 30$, $\text{Initial Inventory}_4 = 200$
- $\text{FDK57}_5$: $r_5 = 120.544$, $\text{Demand}_5 = 10$, $\text{Initial Inventory}_5 = 150$

##### Model

\[
\begin{align*}
\max\quad & 119.144\, x_1 + 119.144\, x_2 + 120.144\, x_3 + 121.244\, x_4 + 120.544\, x_5 \\
\text{s.t.}\quad
& 0 \leq x_1 \leq \min\{200, 30\} = 30 \\
& 0 \leq x_2 \leq \min\{100, 40\} = 40 \\
& 0 \leq x_3 \leq \min\{150, 50\} = 50 \\
& 0 \leq x_4 \leq \min\{200, 30\} = 30 \\
& 0 \leq x_5 \leq \min\{150, 10\} = 10 \\
& x_i \geq 0,\quad \forall i=1,\ldots,5
\end{align*}
\]

##### Summary Table

| $i$         | $r_i$    | Demand | Initial Inventory | Upper Bound for $x_i$ |
|-------------|----------|--------|------------------|-----------------------|
| FDK57$_1$   | 119.144  | 30     | 200              | 30                    |
| FDK57$_2$   | 119.144  | 40     | 100              | 40                    |
| FDK57$_3$   | 120.144  | 50     | 150              | 50                    |
| FDK57$_4$   | 121.244  | 30     | 200              | 30                    |
| FDK57$_5$   | 120.544  | 10     | 150              | 10                    |

##### Complete Model

\[
\begin{align*}
\max\quad & 119.144\, x_1 + 119.144\, x_2 + 120.144\, x_3 + 121.244\, x_4 + 120.544\, x_5 \\
\text{s.t.}\quad
& 0 \leq x_1 \leq 30 \\
& 0 \leq x_2 \leq 40 \\
& 0 \leq x_3 \leq 50 \\
& 0 \leq x_4 \leq 30 \\
& 0 \leq x_5 \leq 10 \\
& x_i \geq 0,\quad i=1,\ldots,5
\end{align*}
\]

where $x_i$ is the fulfilled quantity for each ‘FDK57’ car model $i$ as listed above.