#### Index Sets

- $I$: set of areas (from products.csv, column ProductName)

#### Parameters

- $b_i$: development benefit per unit scale in area $i$ (from products.csv, column Value)
- $w_i$: resource usage per unit scale in area $i$ (from products.csv, column Weight)
- $C$: overall development capacity (from capacity.csv, column Capacity)

#### Decision Variables

- $x_i \geq 0$: scale of development per day in area $i$, $\forall i \in I$

#### Objective

$$
\max \sum_{i \in I} b_i x_i
$$

#### Constraints

1. Overall development capacity:
   $$
   \sum_{i \in I} w_i x_i \leq C
   $$
2. Non-negativity:
   $$
   x_i \geq 0, \quad \forall i \in I
   $$

#### Data Mapping

- products.csv: 
  - ProductName $\rightarrow$ area index set $I$
  - Value $\rightarrow$ benefit parameter $b_i$
  - Weight $\rightarrow$ resource usage parameter $w_i$
- capacity.csv:
  - Capacity $\rightarrow$ overall development capacity $C$