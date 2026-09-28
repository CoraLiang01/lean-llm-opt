#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of drug types (from products.csv, column ProductName)

**Parameters:**
- $v_i$: benefit coefficient for drug type $i \in I$ (from products.csv, column Value)
- $w_i$: weight per unit for drug type $i \in I$ (from products.csv, column Weight)
- $C$: overall inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: integer number of units of drug type $i \in I$ to order daily, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
1. **Overall Inventory Capacity:**
   \[
   \sum_{i \in I} w_i x_i \leq C
   \]
2. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

**Data Mapping:**
- Table: capacity.csv, Column: Capacity $\rightarrow$ parameter $C$
- Table: products.csv, Column: ProductName $\rightarrow$ index set $I$
- Table: products.csv, Column: Value $\rightarrow$ parameter $v_i$
- Table: products.csv, Column: Weight $\rightarrow$ parameter $w_i$