##### Sets and Indices

Let $I = \{1,2,\ldots,100\}$ index the products, with product $i$ corresponding to $P_i$.

Let $R = \{1,2,3\}$ index the resources, corresponding to $R1$, $R2$, $R3$.

##### Parameters (from CSV, source order)

For each product $i$:

- $profit\_per\_unit_i$ = profit per unit of $P_i$
- $r1\_per\_unit_i$ = units of $R1$ consumed per unit of $P_i$
- $r2\_per\_unit_i$ = units of $R2$ consumed per unit of $P_i$
- $r3\_per\_unit_i$ = units of $R3$ consumed per unit of $P_i$
- $upper\_demand\_units_i$ = maximum market demand (units) for $P_i$
- $batch\_size\_units = 10$ (for all $i$)

Resource capacities:

- $capacity_{R1} = 27380.54$
- $capacity_{R2} = 22245.11$
- $capacity_{R3} = 15147.73$

##### Decision Variables

For each product $i$:

- $x_i \in \mathbb{Z}_+$: number of batches of product $i$ to produce (integer, $x_i \geq 0$)

##### Objective

$\max \sum_{i=1}^{100} 10 \cdot x_i \cdot profit\_per\_unit_i$

##### Constraints

For each resource $r$:

- Resource 1: $\sum_{i=1}^{100} 10 \cdot x_i \cdot r1\_per\_unit_i \leq 27380.54$
- Resource 2: $\sum_{i=1}^{100} 10 \cdot x_i \cdot r2\_per\_unit_i \leq 22245.11$
- Resource 3: $\sum_{i=1}^{100} 10 \cdot x_i \cdot r3\_per\_unit_i \leq 15147.73$

For each product $i$:

- Demand upper bound: $10 \cdot x_i \leq upper\_demand\_units_i$

- Integer batches: $x_i \in \mathbb{Z}_+, \quad x_i \geq 0$

##### Full Data Table (source order, first 5 shown for illustration; all 100 included in model)

| $i$ | product | $profit\_per\_unit_i$ | $r1\_per\_unit_i$ | $r2\_per\_unit_i$ | $r3\_per\_unit_i$ | $upper\_demand\_units_i$ | $batch\_size\_units$ |
|---|---|---|---|---|---|---|---|
| 1 | P1 | 6.7 | 2.37 | 0.61 | 2.03 | 317 | 10 |
| 2 | P2 | 10.96 | 4.79 | 2.73 | 0.53 | 106 | 10 |
| 3 | P3 | 9.4 | 3.87 | 1.6 | 0.74 | 386 | 10 |
| 4 | P4 | 11.13 | 3.31 | 2.28 | 2.73 | 441 | 10 |
| 5 | P5 | 10.29 | 1.46 | 3.68 | 1.94 | 63 | 10 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 100 | P100 | 8.84 | 1.25 | 3.23 | 0.53 | 576 | 10 |

##### Complete Model (all coefficients from CSV, source order):

$\max \sum_{i=1}^{100} 10 \cdot x_i \cdot profit\_per\_unit_i$

subject to

$\sum_{i=1}^{100} 10 \cdot x_i \cdot r1\_per\_unit_i \leq 27380.54$

$\sum_{i=1}^{100} 10 \cdot x_i \cdot r2\_per\_unit_i \leq 22245.11$

$\sum_{i=1}^{100} 10 \cdot x_i \cdot r3\_per\_unit_i \leq 15147.73$

$10 \cdot x_i \leq upper\_demand\_units_i \quad \forall i=1,\ldots,100$

$x_i \in \mathbb{Z}_+, \quad x_i \geq 0 \quad \forall i=1,\ldots,100$

##### All coefficients and identifiers (source order):

{
  "products": [
    {"product":"P1","profit_per_unit":6.7,"r1_per_unit":2.37,"r2_per_unit":0.61,"r3_per_unit":2.03,"upper_demand_units":317,"batch_size_units":10},
    {"product":"P2","profit_per_unit":10.96,"r1_per_unit":4.79,"r2_per_unit":2.73,"r3_per_unit":0.53,"upper_demand_units":106,"batch_size_units":10},
    {"product":"P3","profit_per_unit":9.4,"r1_per_unit":3.87,"r2_per_unit":1.6,"r3_per_unit":0.74,"upper_demand_units":386,"batch_size_units":10},
    {"product":"P4","profit_per_unit":11.13,"r1_per_unit":3.31,"r2_per_unit":2.28,"r3_per_unit":2.73,"upper_demand_units":441,"batch_size_units":10},
    {"product":"P5","profit_per_unit":10.29,"r1_per_unit":1.46,"r2_per_unit":3.68,"r3_per_unit":1.94,"upper_demand_units":63,"batch_size_units":10},
    ... (all products P6–P100, in source order, with all coefficients as above)
  ],
  "resource_capacities": {
    "R1": 27380.54,
    "R2": 22245.11,
    "R3": 15147.73
  }
}

##### Variable domains

$x_i \in \mathbb{Z}_+, \quad x_i \geq 0$ (integer, nonnegative, for all $i$)

##### Batch size

All production and constraints are in multiples of $10$ units per batch, as specified by $batch\_size\_units=10$ for all products.