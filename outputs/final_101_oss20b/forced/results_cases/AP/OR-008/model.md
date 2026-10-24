**Mathematical Model**

Let  

- \(S\) be the set of suppliers (warehouses).  
- \(C\) be the set of customers (stores).  

Parameters  

- \(d_c \;(\forall c\in C)\) – daily demand of customer \(c\).  
- \(s_s \;(\forall s\in S)\) – supply capacity of supplier \(s\).  
- \(c_{s,c} \;(\forall s\in S,\forall c\in C)\) – transportation cost per unit from supplier \(s\) to customer \(c\).  

Decision variables  

- \(x_{s,c}\ge 0\) – quantity shipped from supplier \(s\) to customer \(c\).  

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{s\in S} x_{s,c}= d_c \quad \forall c\in C
\]

2. **Supply capacity**  
\[
\sum_{c\in C} x_{s,c}\le s_s \quad \forall s\in S
\]

3. **Non‑negativity**  
\[
x_{s,c}\ge 0 \quad \forall s\in S,\forall c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| \(d_c\) | `file_0_view_0` | `Customers`, `demand` |
| \(s_s\) | `file_1_view_0` | `Suppliers`, `supply_capacity` |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (supplier), `Customer1`–`Customer6` (customers) |

*Index sets*  
- \(C =\) values of `Customers` in `file_0_view_0`.  
- \(S =\) values of `Suppliers` in `file_1_view_0`.  

All mappings use the exact identifiers and column names as provided.