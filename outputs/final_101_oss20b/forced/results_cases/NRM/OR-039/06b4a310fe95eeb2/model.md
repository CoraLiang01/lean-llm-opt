##### Abstract Optimization Model

**Index Sets**  
\[
\begin{aligned}
I &\;=\; \text{set of vehicle types} \\
W &\;=\; \text{set of warehouses}
\end{aligned}
\]

**Parameters**  
\[
\begin{aligned}
b_i &\;=\; \text{benefit coefficient of vehicle type } i \quad (i\in I)\\
C_w &\;=\; \text{capacity limit of warehouse } w \quad (w\in W)
\end{aligned}
\]

**Decision Variables**  
\[
x_{i,w}\;\in\;\mathbb{Z}_{\ge 0}\quad \forall i\in I,\; w\in W
\]
\(x_{i,w}\) is the number of vehicles of type \(i\) stored in warehouse \(w\) per day.

**Objective Function**  
\[
\max \;\sum_{i\in I}\sum_{w\in W} b_i \, x_{i,w}
\]

**Constraints**  

1. **Warehouse Capacity Constraints**  
\[
\sum_{i\in I} x_{i,w} \;\le\; C_w \quad \forall w\in W
\]

2. **Non‑negativity and Integer Constraints**  
\[
x_{i,w}\;\in\;\mathbb{Z}_{\ge 0}\quad \forall i\in I,\; w\in W
\]

---

### Data Mapping

| Source File | Table ID | Column Names |
|-------------|----------|--------------|
| `capacity.csv` | `file_0_view_0` | `Warehouse ID`, `Capacity` |
| `products.csv` | `file_1_view_0` | `ProductName`, `Value` |

*The model uses the benefit coefficient from the `Value` column of the `products.csv` table and the capacity limit from the `Capacity` column of the `capacity.csv` table.*