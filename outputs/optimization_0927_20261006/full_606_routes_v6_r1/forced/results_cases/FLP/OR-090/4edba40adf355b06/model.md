##### Sets and Indices

- $I = \{P1, P2, \ldots, P100\}$: set of products.
- $k \in \{1,2,3\}$: resource indices, corresponding to $R1$, $R2$, $R3$.

##### Parameters

- $profit\_per\_unit_i$: profit per unit for product $i$ (see table below).
- $r1\_per\_unit_i$: units of resource $R1$ consumed per unit of product $i$.
- $r2\_per\_unit_i$: units of resource $R2$ consumed per unit of product $i$.
- $r3\_per\_unit_i$: units of resource $R3$ consumed per unit of product $i$.
- $upper\_demand\_units_i$: maximum market demand (units) for product $i$.
- $batch\_size\_units = 10$: fixed batch size (units per batch).
- $capacity_{R1} = 27380.54$: total available amount of resource $R1$.
- $capacity_{R2} = 22245.11$: total available amount of resource $R2$.
- $capacity_{R3} = 15147.73$: total available amount of resource $R3$.

##### Decision Variables

- $x_i \in \mathbb{Z}_+$: number of batches of product $i$ to produce (integer, $\geq 0$).

##### Objective Function

\[
\max \sum_{i \in I} batch\_size\_units \cdot x_i \cdot profit\_per\_unit_i
\]

##### Constraints

1. **Resource Capacity Constraints:**
   - $R1$: $\sum_{i \in I} batch\_size\_units \cdot x_i \cdot r1\_per\_unit_i \leq capacity_{R1}$
   - $R2$: $\sum_{i \in I} batch\_size\_units \cdot x_i \cdot r2\_per\_unit_i \leq capacity_{R2}$
   - $R3$: $\sum_{i \in I} batch\_size\_units \cdot x_i \cdot r3\_per\_unit_i \leq capacity_{R3}$

2. **Demand Upper Bound Constraints:**
   - For all $i \in I$: $batch\_size\_units \cdot x_i \leq upper\_demand\_units_i$

3. **Batch Integer Constraints:**
   - For all $i \in I$: $x_i \in \mathbb{Z}_+, \ x_i \geq 0$

##### Parameter Table

| product | profit_per_unit | r1_per_unit | r2_per_unit | r3_per_unit | upper_demand_units | batch_size_units |
|---------|----------------|-------------|-------------|-------------|-------------------|-----------------|
| P1      | 6.7            | 2.37        | 0.61        | 2.03        | 317               | 10              |
| P2      | 10.96          | 4.79        | 2.73        | 0.53        | 106               | 10              |
| P3      | 9.4            | 3.87        | 1.6         | 0.74        | 386               | 10              |
| P4      | 11.13          | 3.31        | 2.28        | 2.73        | 441               | 10              |
| P5      | 10.29          | 1.46        | 3.68        | 1.94        | 63                | 10              |
| P6      | 8.06           | 1.46        | 1.37        | 0.32        | 221               | 10              |
| P7      | 6.94           | 1.04        | 1.94        | 0.57        | 441               | 10              |
| P8      | 11.44          | 4.44        | 3.14        | 2.09        | 489               | 10              |
| P9      | 9.13           | 3.32        | 1.3         | 0.31        | 277               | 10              |
| P10     | 7.83           | 3.77        | 0.77        | 0.73        | 121               | 10              |
| ...     | ...            | ...         | ...         | ...         | ...               | ...             |
| P100    | 8.84           | 1.25        | 3.23        | 0.53        | 576               | 10              |

(Full table includes all products P1–P100 as retrieved above.)

##### Resource Capacities

- $capacity_{R1} = 27380.54$
- $capacity_{R2} = 22245.11$
- $capacity_{R3} = 15147.73$

##### Model Summary

\[
\begin{align*}
\max \quad & \sum_{i \in I} 10 \cdot x_i \cdot profit\_per\_unit_i \\
\text{s.t.} \quad
& \sum_{i \in I} 10 \cdot x_i \cdot r1\_per\_unit_i \leq 27380.54 \\
& \sum_{i \in I} 10 \cdot x_i \cdot r2\_per\_unit_i \leq 22245.11 \\
& \sum_{i \in I} 10 \cdot x_i \cdot r3\_per\_unit_i \leq 15147.73 \\
& 10 \cdot x_i \leq upper\_demand\_units_i \quad \forall i \in I \\
& x_i \in \mathbb{Z}_+, \ x_i \geq 0 \quad \forall i \in I
\end{align*}
\]

where all coefficients and bounds are as listed in the parameter table above.