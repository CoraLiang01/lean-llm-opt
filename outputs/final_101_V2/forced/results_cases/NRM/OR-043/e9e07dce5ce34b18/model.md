### Abstract Mathematical Model

#### Index Sets
- $I$: Set of drug products (from products.csv, column ProductName)

#### Parameters
- $v_i$: Benefit per unit of drug $i \in I$ (from products.csv, column Value)
- $w_i$: Stock weight per unit of drug $i \in I$ (from products.csv, column Weight)
- $C$: Overall stock capacity (from capacity.csv, column Capacity)

#### Decision Variables
- $x_i$: Number of units of drug $i \in I$ to order each day, $x_i \geq 0$, integer

#### Objective
\[
\max \sum_{i \in I} v_i x_i
\]

#### Constraints
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

---

### Data Mapping

- Table: capacity.csv, Column: Capacity → Parameter $C$
- Table: products.csv, Column: ProductName → Index set $I$
- Table: products.csv, Column: Value → Parameter $v_i$
- Table: products.csv, Column: Weight → Parameter $w_i$