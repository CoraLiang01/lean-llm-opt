#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of products classified under ‘Aalop’ (indexed by $i$; see Data Mapping).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from column Revenue).
- $d_i$: Demand for product $i$ (from column Demand).
- $s_i$: Initial Inventory for product $i$ (from column Initial Inventory).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** $x_i \leq d_i \quad \forall i \in I$
2. **Inventory limit:** $x_i \leq s_i \quad \forall i \in I$
3. **Nonnegativity and integrality:** $x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

#### Data Mapping

- $I$: All rows in RestaurantSalesreport.csv where Product Name starts with "Aalop".
- $r_i$: RestaurantSalesreport.csv, column Revenue, for product $i$.
- $d_i$: RestaurantSalesreport.csv, column Demand, for product $i$.
- $s_i$: RestaurantSalesreport.csv, column Initial Inventory, for product $i$.

**Table Reference:**  
- Table ID: file_0_view_0  
- Columns: Product Name, Revenue, Demand, Initial Inventory  
- Filter: Product Name prefix "Aalop"  
- Returned rows: 1 (Aalopuri)