**Mathematical Model**

Let  

- \(D=\{A,B,C,D,E,F,G,H,I,J\}\) (heavy‑machinery devices)  
- \(P=\{P1,P2,\dots ,P111\}\) (products)

Parameters  

- \(p_i\) unit profit of product \(i\) \((i\in P)\)  
- \(t_{d,i}\) processing time of product \(i\) on device \(d\) \((d\in D,\;i\in P)\)  
- \(c_d\) monthly operating capacity of device \(d\) \((d\in D)\)

Decision variables  

- \(x_i\ge 0\) production quantity of product \(i\) \((i\in P)\)

\[
\begin{aligned}
\max\;& \sum_{i\in P} p_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in P} t_{d,i}\,x_i \;\le\; c_d , && \forall d\in D \\[4pt]
& x_i \;\ge\; 0 , && \forall i\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Mapping Rule |
|-----------|--------------|--------------|
| \(p_i\) | `file_2_view_0` (`unit_product_profits.csv`) | `Unit_Profit` column → \(p_{\text{Product}}\) |
| \(t_{d,i}\) | `file_0_view_0` (`device_time.csv`) | For each row `Device = d`, column `Pj` → \(t_{d,j}\) |
| \(c_d\) | `file_1_view_0` (`monthly_device_capacity.csv`) | `Monthly_Capacity` column → \(c_{\text{Device}}\) |

All identifiers (`Device`, `Product`) are taken directly from the corresponding tables; no additional enumeration or transformation is performed.