### Mathematical Model

Let $I$ be the set of vehicle types (indexed by $i$), with each $i$ corresponding to a unique ProductName from products.csv.

**Parameters:**
- $v_i$: Value (profit) of vehicle type $i$ (from products.csv, column Value)
- $w_i$: Weight (inventory space required) of vehicle type $i$ (from products.csv, column Weight)
- $C$: Overall inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraint:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

### Data Mapping

- $I$: All records in products.csv, column ProductName (table_id: file_1_view_0, column: ProductName)
- $v_i$: products.csv, column Value (table_id: file_1_view_0, column: Value), keyed by ProductName
- $w_i$: products.csv, column Weight (table_id: file_1_view_0, column: Weight), keyed by ProductName
- $C$: capacity.csv, column Capacity (table_id: file_0_view_0, column: Capacity), single value

All variables, parameters, and constraints are mapped directly to the provided data columns and business identifiers.