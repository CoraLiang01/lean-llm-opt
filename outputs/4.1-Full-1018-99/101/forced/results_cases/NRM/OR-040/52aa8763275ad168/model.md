#### Index Sets

- $I$: set of areas (from products.csv, column ProductName)

#### Parameters

- $b_i$: benefit coefficient for area $i \in I$ (from products.csv, column Value)
- $C$: overall development capacity (from capacity.csv, column Capacity)

#### Decision Variables

- $x_i$: integer, scale of development in area $i$ per day, $x_i \in \mathbb{Z}_{\geq 0}$

#### Objective

$$
\max \sum_{i \in I} b_i x_i
$$

#### Constraints

1. **Overall Capacity Constraint:**
   $$
   \sum_{i \in I} x_i \leq C
   $$

2. **Variable Domain:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$, $b_i$: file_1_view_0 (products.csv), columns ProductName (area identifiers), Value (benefit coefficients)
- $C$: file_0_view_0 (capacity.csv), column Capacity

No additional constraints or data fields are used. All records from both files are included as returned by CSVQA.