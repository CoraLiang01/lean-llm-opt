**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName)

**Parameters:**
- $p_i$: Profit per unit of vehicle $i$ (from products.csv, column Value)
- $w_i$: Inventory weight per unit of vehicle $i$ (from products.csv, column Weight)
- $C$: Total inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day; $x_i \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

| Symbol | Source Table | Column Name         | Description                                 |
|--------|-------------|---------------------|---------------------------------------------|
| $I$    | products.csv| ProductName         | Vehicle types                               |
| $p_i$  | products.csv| Value               | Profit per unit of vehicle $i$              |
| $w_i$  | products.csv| Weight              | Inventory weight per unit of vehicle $i$    |
| $C$    | capacity.csv| Capacity            | Total inventory capacity                    |