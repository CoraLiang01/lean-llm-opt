**Mathematical Model**

Let  

* \(W\) – set of warehouses (indexed by the values in column **Warehouse ID** of *capacity.csv*).  
* \(P\) – set of vehicle types (indexed by the values in column **ProductName** of *products.csv*).  

Parameters  

\[
\begin{aligned}
\text{Cap}_w &\;=\; \text{Capacity of warehouse } w \quad (w\in W)\\
\text{Val}_p &\;=\; \text{Value (benefit coefficient) of product } p \quad (p\in P)
\end{aligned}
\]

Decision variables  

\[
x_{p,w}\;\in\;\mathbb{Z}_{\ge 0}\qquad (p\in P,\;w\in W)
\]

*\(x_{p,w}\)* = number of units of product \(p\) stored in warehouse \(w\).

---

### Objective

\[
\max \;\sum_{p\in P}\;\sum_{w\in W}\;\text{Val}_p \; x_{p,w}
\]

---

### Constraints

1. **Warehouse capacity**

\[
\sum_{p\in P} x_{p,w}\;\le\;\text{Cap}_w \qquad \forall\, w\in W
\]

2. **Non‑negativity & integrality**

\[
x_{p,w}\;\in\;\mathbb{Z}_{\ge 0}\qquad \forall\, p\in P,\;w\in W
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(\text{Cap}_w\) | `file_0_view_0` | `Capacity` | For each row, assign the value of **Capacity** to the warehouse identified by **Warehouse ID**. |
| \(\text{Val}_p\) | `file_1_view_0` | `Value` | For each row, assign the value of **Value** to the product identified by **ProductName**. |

*The `Weight` column in *products.csv* is not used in the current formulation.*