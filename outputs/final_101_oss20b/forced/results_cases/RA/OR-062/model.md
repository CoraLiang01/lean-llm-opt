**Mathematical Model**

Let  

* \(I=\{i\}\) – set of suppliers (rows of *fixed_cost.csv*).  
* \(J=\{j\}\) – set of customers (columns of *transportation_costs.csv*).  

Parameters  

\[
\begin{aligned}
&\text{fixed\_cost}_i &&\text{(fixed cost of opening supplier }i)\\
&\text{demand}_j &&\text{(demand of customer }j)\\
&\text{trans\_cost}_{ij} &&\text{(transportation cost per unit from supplier }i\text{ to customer }j)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_i &\in\{0,1\} &&\text{(supplier }i\text{ is open)}\\
&x_{ij} &\in\mathbb{Z}_{\ge 0} &&\text{(units shipped from }i\text{ to }j)
\end{aligned}
\]

---

**Objective**

\[
\min \;\; \sum_{i\in I}\text{fixed\_cost}_i\,y_i
      \;+\; \sum_{i\in I}\sum_{j\in J}\text{trans\_cost}_{ij}\,x_{ij}
\]

---

**Constraints**

1. **Demand satisfaction** – each customer’s demand must be met  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; \text{demand}_j , \qquad \forall j\in J
   \]

2. **Supplier activation** – shipments can occur only from open suppliers  
   \[
   x_{ij} \;\le\; \text{demand}_j \, y_i , \qquad \forall i\in I,\; \forall j\in J
   \]

3. **Variable domains**  
   \[
   y_i \in \{0,1\}, \quad x_{ij}\in\mathbb{Z}_{\ge 0}
   \]

---

**Data Mapping**

| Parameter | Table ID | Key column | Value column |
|-----------|----------|------------|--------------|
| `fixed_cost_i` | `file_1_view_0` | `Unnamed: 0` (supplier name) | `fixed_costs` |
| `demand_j` | `file_0_view_0` | `Customer` | `demand` |
| `trans_cost_{ij}` | `file_2_view_0` | `Unnamed: 0` (supplier name) | column named after customer `j` (e.g., `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`) |

All index sets \(I\) and \(J\) are derived from the distinct values of the key columns in the respective tables. No numerical values are hard‑coded; the model references the data through the mappings above.