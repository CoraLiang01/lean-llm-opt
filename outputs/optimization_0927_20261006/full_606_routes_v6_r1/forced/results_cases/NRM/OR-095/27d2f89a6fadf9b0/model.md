#### Abstract Optimization Model

**Index Sets:**
- $W$: Set of widgets, $W = \{\text{Widget1}, \ldots, \text{Widget141}\}$
- $R$: Set of resources, $R = \{\text{LaborHours}, \text{MaterialA}, \text{MaterialB}\}$

**Parameters:**
- $a_{w,r}$: Amount of resource $r \in R$ required per unit of widget $w \in W$  
  (Data: product_resources.csv, columns: Product, LaborHours, MaterialA, MaterialB)
- $p_w$: Base profit per unit of widget $w \in W$  
  (Data: product_resources.csv, columns: Product, Profit)
- $L_r$: Monthly limit for resource $r \in R$  
  (Data: resource_limits.csv, columns: Resource, MonthlyLimit)
- $q$: CatalystX byproduct generated per unit of Widget3 ($q = 5$ kg/unit)
- $v$: Sale price per kg of CatalystX ($v = 300$)
- $d$: Disposal cost per kg of unsold CatalystX ($d = 200$)
- $S^{\max}$: Maximum CatalystX sales per month ($S^{\max} = 1500$ kg)

**Decision Variables:**
- $x_w \geq 0$: Number of units of widget $w \in W$ to produce (continuous or integer, as appropriate)
- $S \geq 0$: Amount (kg) of CatalystX sold to partners
- $D \geq 0$: Amount (kg) of CatalystX disposed as hazardous waste

**Objective:**
\[
\max \left\{ \sum_{w \in W} p_w x_w + v S - d D \right\}
\]

**Constraints:**

1. **Resource Limits:**
   \[
   \sum_{w \in W} a_{w,r} x_w \leq L_r \qquad \forall r \in R
   \]

2. **CatalystX Byproduct Balance:**
   \[
   S + D = q \cdot x_{\text{Widget3}}
   \]

3. **CatalystX Sales Cap:**
   \[
   S \leq S^{\max}
   \]

4. **Nonnegativity:**
   \[
   x_w \geq 0 \qquad \forall w \in W
   \]
   \[
   S \geq 0,\quad D \geq 0
   \]

---

#### Data Mapping

- **product_resources.csv**  
  - Widget: index set $W$  
  - LaborHours: $a_{w,\text{LaborHours}}$  
  - MaterialA: $a_{w,\text{MaterialA}}$  
  - MaterialB: $a_{w,\text{MaterialB}}$  
  - Profit: $p_w$

- **resource_limits.csv**  
  - Resource: index set $R$  
  - MonthlyLimit: $L_r$

- **CatalystX byproduct and market/disposal parameters**  
  - $q$, $v$, $d$, $S^{\max}$: scenario constants from user description

---

**Note:**  
- All widgets $w \in W$ are included as returned by the query (Widget1 through Widget141).
- All resource constraints and byproduct handling are modeled as described in the user query.
- No additional constraints or selection logic are imposed beyond those specified.