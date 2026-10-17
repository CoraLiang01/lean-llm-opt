**Mathematical Model**

Let  

- \(S\) – set of suppliers (indices from the *supply_capacity* table)  
- \(C\) – set of customers (indices from the *customer_demand* table)

Parameters  

\[
\begin{aligned}
\text{cap}[s] &\;=\; \text{daily supply capacity of supplier }s,\\
\text{dem}[c] &\;=\; \text{daily demand of customer }c,\\
\text{cost}[s,c] &\;=\; \text{transportation cost per unit from }s\text{ to }c .
\end{aligned}
\]

Decision variables  

\[
x[s,c] \;\ge\; 0 \qquad \forall s\in S,\;c\in C
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}[s,c]\;x[s,c]
\]

Supply‑capacity constraints  

\[
\sum_{c\in C} x[s,c] \;\le\; \text{cap}[s] \qquad \forall s\in S
\]

Demand‑fulfilment constraints  

\[
\sum_{s\in S} x[s,c] \;=\; \text{dem}[c] \qquad \forall c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| \(\text{cap}[s]\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(\text{dem}[c]\) | `file_0_view_0` | `customer` | `demand` |
| \(\text{cost}[s,c]\) | `file_2_view_0` | `Unnamed: 0` | `C1` … `C10` (columns for each customer) |

All indices \(s\) and \(c\) are taken directly from the corresponding key columns of the source tables. No additional data or assumptions are introduced beyond what is supplied.