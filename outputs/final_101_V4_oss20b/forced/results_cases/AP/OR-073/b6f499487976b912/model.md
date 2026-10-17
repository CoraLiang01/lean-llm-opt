**Mathematical model**

Let  

* \(P=\{I,II,III\}\) – products  
* \(E=\{A1,A2,A3,B1,B2,B3,B4\}\) – equipment types  

Decision variables  

\[
\begin{aligned}
x_p &\ge 0 &&\text{(units of product }p\text{ produced)}\\
y_{e,p} &\ge 0 &&\text{(units of product }p\text{ processed on equipment }e)
\end{aligned}
\]

Parameters (all taken from *file_0_view_0*)

\[
\begin{aligned}
c_p &\;=\;\text{Unit price of product }p\\
r_p &\;=\;\text{Raw‑material cost per unit of product }p\\
t_{e,p} &\;=\;\text{Processing time per unit of product }p\text{ on equipment }e\\
a_e &\;=\;\text{Available operating time of equipment }e\\
d_e &\;=\;\text{Equipment cost per hour of equipment }e\\
\delta_{e,p} &=
\begin{cases}
1 &\text{if equipment }e\text{ can process product }p\\
0 &\text{otherwise}
\end{cases}
\end{aligned}
\]

---

### Objective (maximise profit)

\[
\max \;\; \sum_{p\in P}\bigl(c_p-r_p\bigr)x_p
      \;-\;\sum_{e\in E}\sum_{p\in P} d_e\,t_{e,p}\,y_{e,p}
\]

---

### Constraints  

1. **Production balance**

\[
x_p=\sum_{e\in E} y_{e,p}\qquad \forall p\in P
\]

2. **Equipment capacity**

\[
\sum_{p\in P} t_{e,p}\,y_{e,p}\;\le\; a_e \qquad \forall e\in E
\]

3. **Eligibility (disallowed pairs)**  

\[
y_{e,p}=0 \qquad \forall e\in E,\;p\in P\;\text{with }\delta_{e,p}=0
\]

4. **Non‑negativity**

\[
x_p\ge0,\; y_{e,p}\ge0 \qquad \forall p\in P,\;e\in E
\]

---

## Data Mapping (table **file_0_view_0**)

| Parameter | Source column(s) | Notes |
|-----------|------------------|-------|
| \(t_{e,p}\) | “Product I”, “Product II”, “Product III” | Row identified by “Equipment / Cost” = \(e\). Empty cells → \(\delta_{e,p}=0\). |
| \(a_e\) | “Available Equipment Operating Time” | Row identified by “Equipment / Cost” = \(e\). |
| \(d_e\) | “Equipment Cost at Full Load (yuan)” | Row identified by “Equipment / Cost” = \(e\). |
| \(r_p\) | “Raw Material Cost (yuan/unit)” | Row “Raw Material Cost (yuan/unit)”, column for product \(p\). |
| \(c_p\) | “Unit Price (yuan/unit)” | Row “Unit Price (yuan/unit)”, column for product \(p\). |
| \(\delta_{e,p}\) | Implicit from presence of \(t_{e,p}\) | 1 if \(t_{e,p}\) is present, 0 otherwise. |

All indices, parameters, and variables are defined exactly as above; no numerical values are enumerated in the model.