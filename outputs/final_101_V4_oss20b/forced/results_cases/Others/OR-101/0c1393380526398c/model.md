## Mathematical Model

**Indices**  
\(D=\{A,B,C,D,E,F,G,H,I,J\}\) (heavy‑machinery devices)  
\(P=\{P1,P2,\dots ,P111\}\) (products)

**Parameters**  

\[
\begin{aligned}
p_P &\;=\;\text{Unit_Profit of product }P
      &&\text{from table }{\tt file\_2\_view\_0},\;{\tt Unit\_Profit} \\[4pt]
t_{d,P} &\;=\;\text{processing time (hours) of product }P\text{ on device }d
      &&\text{from table }{\tt file\_0\_view\_0},\;{\tt P1\!-\!P111} \\[4pt]
C_d &\;=\;\text{monthly operating capacity of device }d
      &&\text{from table }{\tt file\_1\_view\_0},\;{\tt Monthly\_Capacity}
\end{aligned}
\]

(Only devices \(A,B,I,J\) have capacity data; no capacity constraint is imposed for the other devices.)

**Decision Variables**  

\[
x_P \;\ge\;0 \qquad\text{(continuous production quantity of product }P)
\]

**Objective**  

\[
\max \;\sum_{P\in P} p_P\,x_P
\]

**Constraints**  

For every device \(d\) with a defined capacity \(C_d\):

\[
\sum_{P\in P} t_{d,P}\,x_P \;\le\; C_d
\]

(If a device has no capacity entry, the corresponding constraint is omitted.)

---

## Data Mapping

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(p_P\) | `file_2_view_0` | `Unit_Profit` | One row per product \(P\) |
| \(t_{d,P}\) | `file_0_view_0` | `P1`–`P111` | One row per device \(d\); each column is the time for product \(P\) |
| \(C_d\) | `file_1_view_0` | `Monthly_Capacity` | One row per device \(d\) that has capacity data (A, B, I, J) |

All indices, parameters, and variables are defined exactly as they appear in the source data; no additional values are invented.