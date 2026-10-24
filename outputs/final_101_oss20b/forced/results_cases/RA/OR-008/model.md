**Mathematical Model**

Let  

* \(C=\{c\}\) be the set of customers (stores).  
* \(S=\{s\}\) be the set of suppliers (warehouses).  

Decision variables  

\[
x_{s c}\;\;(\text{units of produce shipped from supplier }s\text{ to customer }c)
\]

---

### Objective  

\[
\min \;\; \sum_{s\in S}\sum_{c\in C} \; \text{cost}_{s c}\; x_{s c}
\]

### Constraints  

1. **Demand satisfaction** – each customer’s demand must be met exactly  

\[
\sum_{s\in S} x_{s c}= \text{demand}_{c}\qquad \forall c\in C
\]

2. **Supply capacity** – a supplier cannot ship more than its available capacity  

\[
\sum_{c\in C} x_{s c}\le \text{supply}_{s}\qquad \forall s\in S
\]

3. **Non‑negativity**  

\[
x_{s c}\ge 0 \qquad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Mapping |
|-----------|----------|-----------|---------|
| \(\text{demand}_{c}\) | `file_0_view_0` | `demand` | `demand[c]` |
| \(\text{supply}_{s}\) | `file_1_view_0` | `supply_capacity` | `supply[s]` |
| \(\text{cost}_{s c}\) | `file_2_view_0` | matrix: rows `Unnamed: 0` (Suppliers), columns `Customer1`–`Customer6` | `cost[s,c]` |

*Index sets*  
- \(C\) is the set of values in column `Customers` of `file_0_view_0`.  
- \(S\) is the set of values in column `Suppliers` of `file_1_view_0`.  

All parameters are taken directly from the supplied CSV files; no additional data are introduced.