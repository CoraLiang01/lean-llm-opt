##### Sets and Indices

Let $I = \{P1, P2, \ldots, P100\}$ be the set of products, indexed by $i$.

Let $K = \{R1, R2, R3\}$ be the set of resources, indexed by $k$.

##### Parameters (from CSV, source order)

For each product $i$:

- $profit\_per\_unit_i$ = profit per unit of product $i$
- $r1\_per\_unit_i$ = units of resource R1 consumed per unit of $i$
- $r2\_per\_unit_i$ = units of resource R2 consumed per unit of $i$
- $r3\_per\_unit_i$ = units of resource R3 consumed per unit of $i$
- $upper\_demand\_units_i$ = maximum market demand (units) for $i$
- $batch\_size\_units_i$ = batch size in units for $i$ (all are 10)

Resource capacities:

- $capacity_{R1} = 27380.54$
- $capacity_{R2} = 22245.11$
- $capacity_{R3} = 15147.73$

##### Decision Variables

For each product $i$:

- $x_i \in \mathbb{Z}_+$: number of batches of product $i$ to produce (integer, $x_i \geq 0$)

##### Objective

$\max \sum_{i \in I} 10 \cdot x_i \cdot profit\_per\_unit_i$

##### Constraints

For each resource $k$:

- R1: $\sum_{i \in I} 10 \cdot x_i \cdot r1\_per\_unit_i \leq 27380.54$
- R2: $\sum_{i \in I} 10 \cdot x_i \cdot r2\_per\_unit_i \leq 22245.11$
- R3: $\sum_{i \in I} 10 \cdot x_i \cdot r3\_per\_unit_i \leq 15147.73$

For each product $i$:

- $10 \cdot x_i \leq upper\_demand\_units_i$

- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

##### Data (source order, all coefficients preserved)

Products (first 5 shown for brevity; all 100 included in model):

| product | profit_per_unit | r1_per_unit | r2_per_unit | r3_per_unit | upper_demand_units | batch_size_units |
|---------|----------------|-------------|-------------|-------------|-------------------|-----------------|
| P1      | 6.7            | 2.37        | 0.61        | 2.03        | 317               | 10              |
| P2      | 10.96          | 4.79        | 2.73        | 0.53        | 106               | 10              |
| P3      | 9.4            | 3.87        | 1.6         | 0.74        | 386               | 10              |
| P4      | 11.13          | 3.31        | 2.28        | 2.73        | 441               | 10              |
| P5      | 10.29          | 1.46        | 3.68        | 1.94        | 63                | 10              |
| ...     | ...            | ...         | ...         | ...         | ...               | ...             |
| P100    | 8.84           | 1.25        | 3.23        | 0.53        | 576               | 10              |

Resource capacities:

| resource | capacity   |
|----------|-----------|
| R1       | 27380.54  |
| R2       | 22245.11  |
| R3       | 15147.73  |

##### Full Model

$\max \sum_{i=1}^{100} 10 \cdot x_i \cdot profit\_per\_unit_i$

subject to

$\sum_{i=1}^{100} 10 \cdot x_i \cdot r1\_per\_unit_i \leq 27380.54$

$\sum_{i=1}^{100} 10 \cdot x_i \cdot r2\_per\_unit_i \leq 22245.11$

$\sum_{i=1}^{100} 10 \cdot x_i \cdot r3\_per\_unit_i \leq 15147.73$

$10 \cdot x_i \leq upper\_demand\_units_i \quad \forall i=1,\ldots,100$

$x_i \in \mathbb{Z}_+ \quad \forall i=1,\ldots,100$

where all coefficients and identifiers are as listed above and in the retrieved data.