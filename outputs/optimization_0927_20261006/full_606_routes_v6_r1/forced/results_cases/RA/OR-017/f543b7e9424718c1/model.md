Let $x_i$ be the number of units of product $i$ (SKU) to fulfill, for each product classified as ‘ZZ’. All variables are nonnegative integers.

##### Objective Function

$$
\max \; 24.38\, x_{\text{ZZ2AO}} + 30.12\, x_{\text{ZZDW7}} + 19.52\, x_{\text{ZZM1A}} + 10.79\, x_{\text{ZZNC5}} + 111.81\, x_{\text{ZZX6K}}
$$

##### Constraints

For each SKU $i$:

- Demand constraint:
  $$
  x_i \leq \text{Demand}_i
  $$
- Inventory constraint:
  $$
  x_i \leq \text{Initial Inventory}_i
  $$
- Nonnegativity and integrality:
  $$
  x_i \in \mathbb{Z}_{\geq 0}
  $$

Explicitly, for each product:

1. For SKU ZZ2AO:
   $$
   x_{\text{ZZ2AO}} \leq 2
   $$
   $$
   x_{\text{ZZ2AO}} \leq 10.0
   $$

2. For SKU ZZDW7:
   $$
   x_{\text{ZZDW7}} \leq 4
   $$
   $$
   x_{\text{ZZDW7}} \leq 20.0
   $$

3. For SKU ZZM1A:
   $$
   x_{\text{ZZM1A}} \leq 82
   $$
   $$
   x_{\text{ZZM1A}} \leq 530.0
   $$

4. For SKU ZZNC5:
   $$
   x_{\text{ZZNC5}} \leq 2
   $$
   $$
   x_{\text{ZZNC5}} \leq 10.0
   $$

5. For SKU ZZX6K:
   $$
   x_{\text{ZZX6K}} \leq 2
   $$
   $$
   x_{\text{ZZX6K}} \leq 10.0
   $$

And for all $i$:
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

##### Decision Variables

- $x_{\text{ZZ2AO}}$: units of SKU ZZ2AO to fulfill
- $x_{\text{ZZDW7}}$: units of SKU ZZDW7 to fulfill
- $x_{\text{ZZM1A}}$: units of SKU ZZM1A to fulfill
- $x_{\text{ZZNC5}}$: units of SKU ZZNC5 to fulfill
- $x_{\text{ZZX6K}}$: units of SKU ZZX6K to fulfill

##### Parameters (from data)

| SKU      | Revenue | Demand | Initial Inventory |
|----------|---------|--------|------------------|
| ZZ2AO    | 24.38   | 2      | 10.0             |
| ZZDW7    | 30.12   | 4      | 20.0             |
| ZZM1A    | 19.52   | 82     | 530.0            |
| ZZNC5    | 10.79   | 2      | 10.0             |
| ZZX6K    | 111.81  | 2      | 10.0             |