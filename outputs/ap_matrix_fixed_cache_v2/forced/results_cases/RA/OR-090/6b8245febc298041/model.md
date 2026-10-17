Let $x_i$ be the number of batches of product $i$ to produce or purchase, for each product $i \in \{\text{P1}, \text{P2}, \ldots, \text{P100}\}$. Each batch is 10 units.

Parameters (from factory_products_100.csv and resources_capacities.csv):

- $\text{profit\_per\_unit}_i$: profit per unit of product $i$
- $\text{r1\_per\_unit}_i$: units of resource R1 consumed per unit of product $i$
- $\text{r2\_per\_unit}_i$: units of resource R2 consumed per unit of product $i$
- $\text{r3\_per\_unit}_i$: units of resource R3 consumed per unit of product $i$
- $\text{upper\_demand\_units}_i$: maximum market demand (units) for product $i$
- $\text{batch\_size\_units}_i = 10$ for all $i$
- $\text{capacity}_{R1} = 27380.54$
- $\text{capacity}_{R2} = 22245.11$
- $\text{capacity}_{R3} = 15147.73$

Model:

Maximize total profit:
$$
\max \sum_{i=1}^{100} 10 \cdot x_i \cdot \text{profit\_per\_unit}_i
$$

Subject to resource constraints:
\[
\sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r1\_per\_unit}_i \leq 27380.54
\]
\[
\sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r2\_per\_unit}_i \leq 22245.11
\]
\[
\sum_{i=1}^{100} 10 \cdot x_i \cdot \text{r3\_per\_unit}_i \leq 15147.73
\]

Subject to demand upper bounds (for each $i$):
\[
10 \cdot x_i \leq \text{upper\_demand\_units}_i \qquad \forall i \in \{\text{P1}, \ldots, \text{P100}\}
\]

Variable domains:
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{\text{P1}, \ldots, \text{P100}\}
\]

Where all coefficients and bounds are as given in the retrieved data:

- For each product $i$ (P1–P100), use the corresponding values for $\text{profit\_per\_unit}_i$, $\text{r1\_per\_unit}_i$, $\text{r2\_per\_unit}_i$, $\text{r3\_per\_unit}_i$, $\text{upper\_demand\_units}_i$, and $\text{batch\_size\_units}_i = 10$.

- Resource capacities are:
    - R1: $27380.54$
    - R2: $22245.11$
    - R3: $15147.73$

All variables, coefficients, and constraints are included as specified in the data and user description.