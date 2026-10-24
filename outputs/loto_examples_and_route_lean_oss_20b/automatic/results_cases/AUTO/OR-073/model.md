**Mathematical Model**

Let  

* \(P=\{I,II,III\}\) – set of products  
* \(A=\{A1,A2\}\) – set of equipment that performs procedure A  
* \(B=\{B1,B2,B3\}\) – set of equipment that performs procedure B  
* \(E=A\cup B\) – all equipment considered  

Parameters (to be read from *file_0_view_0*)  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(t_{i,e}\) | processing time per unit of product \(i\) on equipment \(e\) (h) | column “Product I/II/III” for row “Equipment / Cost” = \(e\) |
| \(a_e\) | available operating time of equipment \(e\) (h) | column “Available Equipment Operating Time” for row “Equipment / Cost” = \(e\) |
| \(c_e\) | equipment operating cost per hour of equipment \(e\) (yuan) | column “Equipment Cost at Full Load (yuan)” for row “Equipment / Cost” = \(e\) |
| \(r_i\) | raw‑material cost per unit of product \(i\) (yuan) | column “Product I/II/III” for row “Equipment / Cost” = “Raw Material Cost (yuan/unit)” |
| \(p_i\) | unit selling price of product \(i\) (yuan) | column “Product I/II/III” for row “Equipment / Cost” = “Unit Price (yuan/unit)” |
| \(\delta_{i,e}\) | 1 if product \(i\) can be processed on equipment \(e\), 0 otherwise | \(\delta_{i,e}=1\) iff \(t_{i,e}>0\) |

Decision variables  

* \(q_i \ge 0\) – production quantity of product \(i\) (continuous)  
* \(x_{i,e} \ge 0\) – amount of product \(i\) processed on equipment \(e\) (continuous)

Large constant \(M\) (e.g., \(10^6\)).

**Objective**

\[
\max \;\sum_{i\in P}\bigl(p_i-r_i\bigr)\,q_i \;-\;\sum_{e\in E} c_e \sum_{i\in P} t_{i,e}\,x_{i,e}
\]

**Constraints**

1. **Procedure A balance**  
   \[
   \sum_{e\in A} x_{i,e}=q_i \qquad \forall i\in P
   \]

2. **Procedure B balance**  
   \[
   \sum_{e\in B} x_{i,e}=q_i \qquad \forall i\in P
   \]

3. **Equipment capacity**  
   \[
   \sum_{i\in P} t_{i,e}\,x_{i,e}\;\le\; a_e \qquad \forall e\in E
   \]

4. **Allowed‑equipment restriction**  
   \[
   x_{i,e}\;\le\; M\,\delta_{i,e} \qquad \forall i\in P,\; e\in E
   \]

5. **Non‑negativity**  
   \[
   q_i\ge0,\; x_{i,e}\ge0 \qquad \forall i\in P,\; e\in E
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column | Row Identifier | Mapping |
|-----------|----------|--------|----------------|---------|
| \(t_{i,e}\) | file_0_view_0 | “Product I”, “Product II”, “Product III” | “Equipment / Cost” = \(e\) | value |
| \(a_e\) | file_0_view_0 | “Available Equipment Operating Time” | “Equipment / Cost” = \(e\) | value |
| \(c_e\) | file_0_view_0 | “Equipment Cost at Full Load (yuan)” | “Equipment / Cost” = \(e\) | value |
| \(r_i\) | file_0_view_0 | “Product I”, “Product II”, “Product III” | “Equipment / Cost” = “Raw Material Cost (yuan/unit)” | value |
| \(p_i\) | file_0_view_0 | “Product I”, “Product II”, “Product III” | “Equipment / Cost” = “Unit Price (yuan/unit)” | value |
| \(\delta_{i,e}\) | file_0_view_0 | – | – | 1 if \(t_{i,e}>0\), else 0 |

All other parameters are derived from these mappings.