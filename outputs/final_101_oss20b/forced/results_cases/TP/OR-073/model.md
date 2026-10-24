**Mathematical model**

Let  

* \(P=\{I,II,III\}\) – products  
* \(A=\{A1,A2,A3\}\) – equipment that performs procedure A  
* \(B=\{B1,B2,B3,B4\}\) – equipment that performs procedure B  
* \(E=A\cup B\) – all equipment  
* \(\mathcal{A}\subseteq P\times E\) – allowed product‑equipment pairs (i.e. the table entry for the pair is non‑empty)

Decision variables  
\[
x_{p,e}\;\ge 0\qquad (p\in P,\;e\in E,\;(p,e)\in\mathcal{A})
\]
= number of units of product \(p\) processed on equipment \(e\).

Parameters  

* \(t_{p,e}\) – processing time per unit of product \(p\) on equipment \(e\)  
* \(a_e\) – available operating time of equipment \(e\)  
* \(c_e\) – operating cost per hour of equipment \(e\)  
* \(r_p\) – raw‑material cost per unit of product \(p\)  
* \(s_p\) – selling price per unit of product \(p\)

---

### Objective
\[
\max\;\sum_{p\in P}\bigl(s_p-r_p\bigr)\sum_{e\in A}x_{p,e}
-\sum_{e\in E}c_e\sum_{p\in P}t_{p,e}\,x_{p,e}
\]

### Constraints  

1. **Production consistency** – each product must receive the same amount of work in both procedures  
\[
\sum_{e\in A}x_{p,e}\;=\;\sum_{e\in B}x_{p,e}\qquad\forall\,p\in P
\]

2. **Equipment capacity** – total time used on each equipment cannot exceed its available time  
\[
\sum_{p\in P}t_{p,e}\,x_{p,e}\;\le\;a_e\qquad\forall\,e\in E
\]

3. **Non‑negativity** – already stated in the variable definition.

---

## Data‑mapping rules

| Parameter | Source table | Row identifier | Column(s) | Notes |
|-----------|--------------|----------------|-----------|-------|
| \(t_{p,e}\) | `file_0_view_0` | `Equipment / Cost = e` | `Product I`, `Product II`, `Product III` | Use the numeric value; if the cell is empty, the pair \((p,e)\) is not in \(\mathcal{A}\). |
| \(a_e\) | `file_0_view_0` | `Equipment / Cost = e` | `Available Equipment Operating Time` | Numeric value. |
| \(c_e\) | `file_0_view_0` | `Equipment / Cost = e` | `Equipment Cost at Full Load (yuan)` | Numeric value. |
| \(r_p\) | `file_0_view_0` | `Equipment / Cost = "Raw Material Cost (yuan/unit)"` | `Product I`, `Product II`, `Product III` | Numeric value. |
| \(s_p\) | `file_0_view_0` | `Equipment / Cost = "Unit Price (yuan/unit)"` | `Product I`, `Product II`, `Product III` | Numeric value. |

**Index sets**

| Set | Definition | Source |
|-----|------------|--------|
| \(P\) | \{I, II, III\} | Derived from column headers “Product I”, “Product II”, “Product III”. |
| \(A\) | \{A1, A2, A3\} | Rows where `Equipment / Cost` starts with “A”. |
| \(B\) | \{B1, B2, B3, B4\} | Rows where `Equipment / Cost` starts with “B”. |
| \(E\) | \(A\cup B\) | Union of \(A\) and \(B\). |
| \(\mathcal{A}\) | \{(p,e) | \(t_{p,e}\) is defined\} | Determined by non‑empty cells in the table. |

All parameters and sets are obtained directly from the single source table `file_0_view_0`; no additional data are introduced.