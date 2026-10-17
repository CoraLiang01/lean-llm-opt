#### Index Sets

- $S$: set of sections (from file_0_view_0, column SectionID)
- $P$: set of products (from file_1_view_0, column ProductName)

#### Parameters

- $C_s$: capacity of section $s \in S$ (from file_0_view_0, column Capacity)
- $v_p$: price (value) of product $p \in P$ (from file_1_view_0, column Value)
- $w_p$: shelf space requirement (weight) of product $p \in P$ (from file_1_view_0, column Weight)

#### Decision Variables

- $x_{sp}$: integer number of units of product $p$ to stock in section $s$, $\forall s \in S, p \in P$

#### Objective

$$
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
$$

#### Constraints

1. **Section Capacity Constraints**  
   For each section $s \in S$:
   $$
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s
   $$

2. **Integrality Constraints**  
   $$
   x_{sp} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S, p \in P
   $$

---

#### Data Mapping

- $S$ (sections), $C_s$: from table_id file_0_view_0, columns SectionID and Capacity, in "capacity.csv"
- $P$ (products), $v_p$, $w_p$: from table_id file_1_view_0, columns ProductName, Value, and Weight, in "products.csv"