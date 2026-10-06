**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of products (widgets), indexed by $i$.  
  [All Product values from file_1_view_0, column Product]
- $R$: Set of resources, indexed by $r$.  
  [All Resource values from file_0_view_0, column Resource]

**Parameters**
- $a_{i,r}$: Amount of resource $r$ consumed per unit of product $i$.  
  [file_1_view_0: columns LaborHours, MaterialA, MaterialB]
- $p_i$: Base profit per unit of product $i$.  
  [file_1_view_0: column Profit]
- $L_r$: Monthly limit for resource $r$.  
  [file_0_view_0: column MonthlyLimit]
- $q$: CatalystX generated per unit of Widget3 (kg/unit).  
  [Query: $q = 5$]
- $v$: Sale price per kg of CatalystX.  
  [Query: $v = 300$]
- $d$: Disposal cost per kg of unsold CatalystX.  
  [Query: $d = 200$]
- $S^{\max}$: Maximum CatalystX sales per month (kg).  
  [Query: $S^{\max} = 1500$]
- $i^*$: Index of Widget3 in $I$.  
  [file_1_view_0: Product = "Widget3"]

**Decision Variables**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to produce.
- $S \geq 0$: Amount (kg) of CatalystX sold (continuous).

**Objective Function**
\[
\max \left\{ \sum_{i \in I} p_i x_i + v S - d \left[q x_{i^*} - S\right] \right\}
\]
where $q x_{i^*}$ is the total CatalystX generated, $S$ is the amount sold, and $q x_{i^*} - S$ is the amount disposed.

**Constraints**
1. **Resource Capacity Constraints**  
  For each $r \in R$:
\[
\sum_{i \in I} a_{i,r} x_i \leq L_r
\]
  where $a_{i,r}$ is the value from file_1_view_0 for product $i$ and resource $r$.

2. **CatalystX Sales Limit**
\[
0 \leq S \leq \min\{q x_{i^*},\ S^{\max}\}
\]
  That is, CatalystX sold cannot exceed what is produced or the market cap.

3. **Nonnegativity and Integrality**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]
\[
S \geq 0
\]

---

**Data Mapping**

- $I$: file_1_view_0, column Product
- $R$: file_0_view_0, column Resource
- $a_{i,\text{LaborHours}}$: file_1_view_0, column LaborHours, keyed by Product
- $a_{i,\text{MaterialA}}$: file_1_view_0, column MaterialA, keyed by Product
- $a_{i,\text{MaterialB}}$: file_1_view_0, column MaterialB, keyed by Product
- $p_i$: file_1_view_0, column Profit, keyed by Product
- $L_{\text{LaborHours}}$: file_0_view_0, column MonthlyLimit, Resource = LaborHours
- $L_{\text{MaterialA}}$: file_0_view_0, column MonthlyLimit, Resource = MaterialA
- $L_{\text{MaterialB}}$: file_0_view_0, column MonthlyLimit, Resource = MaterialB
- $i^*$: file_1_view_0, Product = "Widget3"
- $q = 5$, $v = 300$, $d = 200$, $S^{\max} = 1500$: (from query description)

---

**Summary**

- Maximize total profit from widget production, CatalystX sales, and disposal costs.
- Subject to labor, Material A, and Material B monthly limits.
- CatalystX sales cannot exceed production or market cap.
- All widget production variables are nonnegative integers; CatalystX sales is continuous and nonnegative.