**Abstract Mathematical Model**

---

### Index Sets  
\[
\begin{aligned}
\mathcal{P} &= \{P_1, P_2, \dots , P_{111}\} \quad &&\text{(products)}\\
\mathcal{D} &= \{A, B, C, D, E, F, G, H, I, J\} \quad &&\text{(devices)}
\end{aligned}
\]

### Parameters  
\[
\begin{aligned}
p_i &\quad &&\text{unit profit of product } i, && i\in\mathcal{P}\\
t_{d,i} &\quad &&\text{processing time of product } i \text{ on device } d, && d\in\mathcal{D},\, i\in\mathcal{P}\\
C_d &\quad &&\text{monthly operating capacity of device } d, && d\in\mathcal{D}
\end{aligned}
\]

### Decision Variables  
\[
x_i \quad &&\text{production quantity of product } i, && i\in\mathcal{P}
\]

\[
x_i \in \mathbb{R}_{+}\quad &&\text{(continuous, non‑negative)}
\]

### Objective Function  
\[
\max \;\; \sum_{i\in\mathcal{P}} p_i \, x_i
\]

### Constraints  

1. **Device Capacity Constraints**  
\[
\sum_{i\in\mathcal{P}} t_{d,i}\, x_i \;\le\; C_d , \qquad \forall d\in\mathcal{D}
\]

2. **Non‑negativity**  
\[
x_i \;\ge\; 0 , \qquad \forall i\in\mathcal{P}
\]

---

## Data Mapping

| Source | Table ID | Exact Column Names |
|--------|----------|--------------------|
| Unit profit data | `file_2_view_0` | `Product`, `Unit_Profit` |
| Device processing times | `file_0_view_0` | `Device`, `P1`, `P2`, …, `P111` |
| Device monthly capacities | `file_1_view_0` | `Device`, `Monthly_Capacity` |

*The mapping shows how each parameter in the model is obtained from the corresponding CSV file and column.*