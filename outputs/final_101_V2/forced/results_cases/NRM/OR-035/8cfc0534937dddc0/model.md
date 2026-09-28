#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of bread types (from products.csv, column ProductName)

**Parameters:**

- $v_i$ : Expected profit per unit of bread type $i$ (from products.csv, column Value)
- $w_i$ : Storage weight per unit of bread type $i$ (from products.csv, column Weight)
- $C$ : Total storage capacity (from capacity.csv, column Capacity)

**Decision Variables:**

- $x_i$ : Number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**

$$
\max \sum_{i \in I} v_i x_i
$$

**Constraints:**

1. **Storage Capacity Constraint:**
   $$
   \sum_{i \in I} w_i x_i \leq C
   $$

2. **Integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- Table: capacity.csv
  - Column: Capacity $\rightarrow$ $C$
- Table: products.csv
  - Column: ProductName $\rightarrow$ $I$
  - Column: Value $\rightarrow$ $v_i$
  - Column: Weight $\rightarrow$ $w_i$