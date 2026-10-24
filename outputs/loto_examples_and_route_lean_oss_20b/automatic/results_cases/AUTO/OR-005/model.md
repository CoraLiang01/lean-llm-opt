**Mathematical Model**

Let  

* \(S=\{s_1,\dots ,s_8\}\) be the set of suppliers (distribution centers).  
* \(C=\{c_1,\dots ,c_4\}\) be the set of customers for which transportation‑cost data are available.  

Parameters  

\[
\begin{aligned}
\text{demand}_{c} &\quad\text{(units required by customer }c),\\
\text{supply\_capacity}_{s} &\quad\text{(maximum units that supplier }s\text{ can ship}),\\
\text{cost}_{s,c} &\quad\text{(transportation cost per unit from supplier }s\text{ to customer }c).
\end{aligned}
\]

Decision variables  

\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\qquad\forall s\in S,\;c\in C
\]

---

### Objective
\[
\min \;\sum_{s\in S}\;\sum_{c\in C}\;\text{cost}_{s,c}\;x_{s,c}
\]

### Constraints  

**Supply capacity (per supplier)**  
\[
\sum_{c\in C} x_{s,c}\;\le\;\text{supply\_capacity}_{s}\qquad\forall s\in S
\]

**Demand fulfillment (per customer)**  
\[
\sum_{s\in S} x_{s,c}\;\ge\;\text{demand}_{c}\qquad\forall c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column(s) |
|-----------|--------------|------------|-----------------|
| \(\text{supply\_capacity}_{s}\) | `file_1_view_0` | `Supplier` | `supply_capacity` |
| \(\text{demand}_{c}\) | `file_0_view_0` | `Customers` | `demand` |
| \(\text{cost}_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (supplier) | `demand1`, `demand2`, `demand3`, `demand4` (customer columns) |

*All indices and values are taken directly from the corresponding rows of the source tables; no additional data are invented.*