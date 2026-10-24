**Mathematical Model**

Let  

- \(S=\{s\}\) be the set of suppliers (IDs from *file_1_view_0* “Unnamed: 0”).  
- \(C=\{c\}\) be the set of customers (IDs from *file_0_view_0* “customer”).  

Parameters  

\[
\begin{aligned}
\text{supply\_capacity}[s] &\;=\; \text{value of column ``supply_capacity'' in file\_1\_view\_0 for supplier }s,\\
\text{demand}[c] &\;=\; \text{value of column ``demand'' in file\_0\_view\_0 for customer }c,\\
\text{cost}[s,c] &\;=\; \text{value of column ``c'' in file\_2\_view\_0 for supplier }s.
\end{aligned}
\]

Decision variables  

\[
x_{s,c}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}[s,c]\;x_{s,c}
\]

Supply‑capacity constraints  

\[
\sum_{c\in C} x_{s,c}\;\le\;\text{supply\_capacity}[s] \qquad \forall s\in S
\]

Demand‑satisfaction constraints  

\[
\sum_{s\in S} x_{s,c}\;\ge\;\text{demand}[c] \qquad \forall c\in C
\]

Non‑negativity and integrality  

\[
x_{s,c}\;\ge\;0,\; x_{s,c}\in\mathbb{Z}\qquad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| `supply_capacity` | `file_1_view_0` | `supply_capacity` (indexed by `Unnamed: 0`) |
| `demand` | `file_0_view_0` | `demand` (indexed by `customer`) |
| `cost` | `file_2_view_0` | `C1`–`C10` (indexed by `Unnamed: 0`) |

All indices and values are taken directly from the supplied CSV observations.