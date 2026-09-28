Let $x_i$ be the number of batches of product $i$ to produce or purchase, for $i \in \{\text{P1}, \text{P2}, \ldots, \text{P100}\}$. Each batch is 10 units.

Maximize total profit:
$$
\max \sum_{i=1}^{100} 10 \cdot \text{profit\_per\_unit}_i \cdot x_i
$$

Subject to resource constraints:
\[
\sum_{i=1}^{100} 10 \cdot \text{r1\_per\_unit}_i \cdot x_i \leq 27380.54
\]
\[
\sum_{i=1}^{100} 10 \cdot \text{r2\_per\_unit}_i \cdot x_i \leq 22245.11
\]
\[
\sum_{i=1}^{100} 10 \cdot \text{r3\_per\_unit}_i \cdot x_i \leq 15147.73
\]

Demand upper bound for each product:
\[
10 x_i \leq \text{upper\_demand\_units}_i \qquad \forall i \in \{\text{P1}, \ldots, \text{P100}\}
\]

Integrality and nonnegativity:
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{\text{P1}, \ldots, \text{P100}\}
\]

Where the coefficients for each product $i$ are as follows (from factory_products_100.csv):

| product | profit_per_unit | r1_per_unit | r2_per_unit | r3_per_unit | upper_demand_units | batch_size_units |
|---------|----------------|-------------|-------------|-------------|--------------------|------------------|
| P1      | 6.7            | 2.37        | 0.61        | 2.03        | 317                | 10               |
| P2      | 10.96          | 4.79        | 2.73        | 0.53        | 106                | 10               |
| P3      | 9.4            | 3.87        | 1.6         | 0.74        | 386                | 10               |
| P4      | 11.13          | 3.31        | 2.28        | 2.73        | 441                | 10               |
| P5      | 10.29          | 1.46        | 3.68        | 1.94        | 63                 | 10               |
| P6      | 8.06           | 1.46        | 1.37        | 0.32        | 221                | 10               |
| P7      | 6.94           | 1.04        | 1.94        | 0.57        | 441                | 10               |
| P8      | 11.44          | 4.44        | 3.14        | 2.09        | 489                | 10               |
| P9      | 9.13           | 3.32        | 1.3         | 0.31        | 277                | 10               |
| P10     | 7.83           | 3.77        | 0.77        | 0.73        | 121                | 10               |
| ...     | ...            | ...         | ...         | ...         | ...                | ...              |
| P100    | 8.84           | 1.25        | 3.23        | 0.53        | 576                | 10               |

Resource capacities (from resources_capacities.csv):

- $R1$: 27380.54
- $R2$: 22245.11
- $R3$: 15147.73

All variables $x_i$ are nonnegative integers.