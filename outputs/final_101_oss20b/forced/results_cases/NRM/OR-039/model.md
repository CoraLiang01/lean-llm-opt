**Symbolic Model**

Let  

* \(W\) – set of warehouses (indexed by the values in *Warehouse ID*).  
* \(P\) – set of vehicle types (indexed by the values in *ProductName*).  

Parameters  

\[
\begin{aligned}
\text{Capacity}[w] &\quad\text{(maximum units that warehouse }w\text{ can hold)}\\
\text{Value}[p] &\quad\text{(benefit coefficient of vehicle type }p\text{)}
\end{aligned}
\]

Decision variables  

\[
x_{w,p}\in \mathbb{Z}_{\ge 0}\qquad \forall\,w\in W,\;p\in P
\]

Objective  

\[
\max \;\sum_{w\in W}\sum_{p\in P}\text{Value}[p]\;x_{w,p}
\]

Constraints  

1. **Warehouse capacity**  
\[
\sum_{p\in P} x_{w,p}\;\le\;\text{Capacity}[w]\qquad \forall\,w\in W
\]

2. **Non‑negativity** (implicit in the variable domain)  

\[
x_{w,p}\ge 0,\;x_{w,p}\in\mathbb{Z}\qquad \forall\,w\in W,\;p\in P
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Capacity}[w]\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}[p]\) | `file_1_view_0` | `Value` |

*Index sets*  
- \(W\) is the set of distinct values in the `Warehouse ID` column of `file_0_view_0`.  
- \(P\) is the set of distinct values in the `ProductName` column of `file_1_view_0`.  

All mappings use the exact table identifiers and column names provided in the observation. No numerical values are enumerated in the model.