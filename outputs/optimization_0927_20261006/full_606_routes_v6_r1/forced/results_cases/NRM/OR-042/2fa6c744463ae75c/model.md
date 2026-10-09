#### Index Sets

- $I$: set of drug types (from products.csv, column ProductName)

#### Parameters

- $v_i$: benefit coefficient of drug type $i$ (from products.csv, column Value)
- $w_i$: weight per unit of drug type $i$ (from products.csv, column Weight)
- $C$: overall inventory capacity (from capacity.csv, column Capacity)

#### Decision Variables

- $x_i$: integer number of units of drug type $i$ to order daily, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

#### Objective

$$
\max \sum_{i \in I} v_i x_i
$$

#### Constraints

1. **Overall Capacity Constraint:**
   $$
   \sum_{i \in I} w_i x_i \leq C
   $$

2. **Nonnegativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$, $v_i$, $w_i$ are defined by all rows of table_id: file_1_view_0, columns: ProductName, Value, Weight (source: products.csv, no filter).
- $C$ is defined by table_id: file_0_view_0, column: Capacity (source: capacity.csv, no filter).