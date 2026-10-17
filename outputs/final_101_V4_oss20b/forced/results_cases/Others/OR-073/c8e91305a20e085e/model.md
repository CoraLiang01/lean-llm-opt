**Mathematical Model**

Let  

* \(P=\{I,II,III\}\) – products  
* \(E_A=\{A1,A2\}\) – equipment that performs procedure A  
* \(E_B=\{B1,B2,B3\}\) – equipment that performs procedure B  

Decision variables  

\[
\begin{aligned}
x_p &\ge 0 &&\text{(units of product }p\text{ produced)}\\
a_{p,e} &\ge 0 &&\text{(units of product }p\text{ processed on equipment }e\in E_A\text{ in procedure A)}\\
b_{p,e} &\ge 0 &&\text{(units of product }p\text{ processed on equipment }e\in E_B\text{ in procedure B)}
\end{aligned}
\]

Parameters (all taken from **file_0_view_0**)

* \(t^A_{p,e}\) – processing time per unit of product \(p\) on equipment \(e\in E_A\)  
  (values from columns “Product I”, “Product II”, “Product III” in rows for A1, A2; blank entries mean the combination is not allowed).  
* \(t^B_{p,e}\) – processing time per unit of product \(p\) on equipment \(e\in E_B\)  
  (values from the same columns in rows for B1, B2, B3).  
* \(h_e\) – equipment cost per hour for equipment \(e\)  
  (column “Equipment Cost at Full Load (yuan)”).  
* \(T_e\) – available operating time for equipment \(e\)  
  (column “Available Equipment Operating Time”).  
* \(c_p\) – raw‑material cost per unit of product \(p\)  
  (row “Raw Material Cost (yuan/unit)”).  
* \(p_p\) – selling price per unit of product \(p\)  
  (row “Unit Price (yuan/unit)”).  

**Objective**

\[
\max \;\sum_{p\in P}\bigl(p_p-c_p\bigr)x_p
      \;-\;\sum_{e\in E_A}h_e\sum_{p\in P}a_{p,e}\,t^A_{p,e}
      \;-\;\sum_{e\in E_B}h_e\sum_{p\in P}b_{p,e}\,t^B_{p,e}
\]

**Constraints**

1. **Assignment of production to procedures**

\[
\sum_{e\in E_A} a_{p,e}=x_p \qquad \forall p\in P
\]
\[
\sum_{e\in E_B} b_{p,e}=x_p \qquad \forall p\in P
\]

2. **Equipment capacity**

\[
\sum_{p\in P} a_{p,e}\,t^A_{p,e}\;\le\;T_e \qquad \forall e\in E_A
\]
\[
\sum_{p\in P} b_{p,e}\,t^B_{p,e}\;\le\;T_e \qquad \forall e\in E_B
\]

3. **Product‑equipment compatibility**  
(variables corresponding to blank entries in the data are fixed to zero)

\[
\begin{aligned}
& a_{III,A1}=0, \\
& b_{III,B1}=0,\; b_{III,B3}=0,\\
& b_{II,B2}=0,\; b_{II,B3}=0.
\end{aligned}
\]

4. **Non‑negativity**

\[
x_p,\;a_{p,e},\;b_{p,e}\;\ge\;0 \qquad \forall p\in P,\;e\in E_A\cup E_B
\]

---

### Data Mapping

| Parameter | Source | Table ID | Column(s) | Notes |
|-----------|--------|----------|-----------|-------|
| \(t^A_{p,e}\) | Processing time for procedure A | file_0_view_0 | “Product I”, “Product II”, “Product III” (rows A1, A2) | Blank → not allowed |
| \(t^B_{p,e}\) | Processing time for procedure B | file_0_view_0 | “Product I”, “Product II”, “Product III” (rows B1, B2, B3) | Blank → not allowed |
| \(h_e\) | Equipment cost per hour | file_0_view_0 | “Equipment Cost at Full Load (yuan)” | For all equipment rows |
| \(T_e\) | Available operating time | file_0_view_0 | “Available Equipment Operating Time” | For all equipment rows |
| \(c_p\) | Raw‑material cost per unit | file_0_view_0 | “Raw Material Cost (yuan/unit)” | Row 7 |
| \(p_p\) | Selling price per unit | file_0_view_0 | “Unit Price (yuan/unit)” | Row 8 |

All indices \(p\) and \(e\) are taken from the non‑blank entries of the corresponding rows/columns. The model uses only the equipment types specified in the problem statement (A1, A2, B1, B2, B3); other equipment rows are ignored.