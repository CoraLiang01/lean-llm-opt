**Symbolic Model**

Let  

* \(P=\{I,II,III\}\) – set of products  
* \(A=\{A1,A2,A3\}\) – equipment that can perform procedure A  
* \(B=\{B1,B2,B3,B4\}\) – equipment that can perform procedure B  

Parameters (all non‑negative)

| Symbol | Meaning | Data source |
|--------|---------|-------------|
| \(t_{p,a}\) | processing time (hours per unit) of product \(p\) on A‑equipment \(a\) | table file_0_view_0, row “Equipment / Cost” = \(a\), column “Product \(p\)” |
| \(t_{p,b}\) | processing time (hours per unit) of product \(p\) on B‑equipment \(b\) | table file_0_view_0, row “Equipment / Cost” = \(b\), column “Product \(p\)” |
| \(f_{p,a}\) | feasibility flag (1 if \(t_{p,a}\) is defined, 0 otherwise) | same cells as \(t_{p,a}\) |
| \(f_{p,b}\) | feasibility flag (1 if \(t_{p,b}\) is defined, 0 otherwise) | same cells as \(t_{p,b}\) |
| \(T_a\) | available operating time (hours) of A‑equipment \(a\) | table file_0_view_0, column “Available Equipment Operating Time” |
| \(T_b\) | available operating time (hours) of B‑equipment \(b\) | same column |
| \(c_a\) | equipment cost per hour of A‑equipment \(a\) | table file_0_view_0, column “Equipment Cost at Full Load (yuan)” |
| \(c_b\) | equipment cost per hour of B‑equipment \(b\) | same column |
| \(r_p\) | raw‑material cost per unit of product \(p\) | table file_0_view_0, row “Raw Material Cost (yuan/unit)”, column “Product \(p\)” |
| \(p_p\) | unit selling price of product \(p\) | table file_0_view_0, row “Unit Price (yuan/unit)”, column “Product \(p\)” |

Decision variables  

* \(x_{p,a}\ge 0\) – units of product \(p\) processed on A‑equipment \(a\)  
* \(x_{p,b}\ge 0\) – units of product \(p\) processed on B‑equipment \(b\)  
* \(y_p\ge 0\) – total units of product \(p\) produced (must equal the amount processed on each procedure)

Auxiliary constant  

* \(M\) – a sufficiently large constant (e.g., larger than any feasible production quantity)

---

### Objective (maximize profit)

\[
\max \;\; \sum_{p\in P}\bigl(p_p-r_p\bigr)\,y_p
\;-\;
\sum_{a\in A}\sum_{p\in P} c_a\,t_{p,a}\,x_{p,a}
\;-\;
\sum_{b\in B}\sum_{p\in P} c_b\,t_{p,b}\,x_{p,b}
\]

---

### Constraints  

1. **Procedure‑A production balance**  
   \[
   \sum_{a\in A} x_{p,a}=y_p \quad \forall p\in P
   \]

2. **Procedure‑B production balance**  
   \[
   \sum_{b\in B} x_{p,b}=y_p \quad \forall p\in P
   \]

3. **Equipment‑time limits (A)**  
   \[
   \sum_{p\in P} t_{p,a}\,x_{p,a}\;\le\;T_a \quad \forall a\in A
   \]

4. **Equipment‑time limits (B)**  
   \[
   \sum_{p\in P} t_{p,b}\,x_{p,b}\;\le\;T_b \quad \forall b\in B
   \]

5. **Feasibility of processing on A‑equipment**  
   \[
   x_{p,a}\;\le\;M\,f_{p,a}\quad \forall p\in P,\;a\in A
   \]

6. **Feasibility of processing on B‑equipment**  
   \[
   x_{p,b}\;\le\;M\,f_{p,b}\quad \forall p\in P,\;b\in B
   \]

7. **Non‑negativity**  
   \[
   x_{p,a}\ge0,\;x_{p,b}\ge0,\;y_p\ge0 \quad \forall p,a,b
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column(s) | Row Identifier |
|-----------|----------|-----------|----------------|
| \(t_{p,a}\) | file_0_view_0 | “Product \(p\)” | “Equipment / Cost” = \(a\) |
| \(t_{p,b}\) | file_0_view_0 | “Product \(p\)” | “Equipment / Cost” = \(b\) |
| \(f_{p,a}\) | file_0_view_0 | same as \(t_{p,a}\) | same |
| \(f_{p,b}\) | file_0_view_0 | same as \(t_{p,b}\) | same |
| \(T_a\) | file_0_view_0 | “Available Equipment Operating Time” | “Equipment / Cost” = \(a\) |
| \(T_b\) | file_0_view_0 | same column | “Equipment / Cost” = \(b\) |
| \(c_a\) | file_0_view_0 | “Equipment Cost at Full Load (yuan)” | “Equipment / Cost” = \(a\) |
| \(c_b\) | file_0_view_0 | same column | “Equipment / Cost” = \(b\) |
| \(r_p\) | file_0_view_0 | “Product \(p\)” | “Equipment / Cost” = “Raw Material Cost (yuan/unit)” |
| \(p_p\) | file_0_view_0 | “Product \(p\)” | “Equipment / Cost” = “Unit Price (yuan/unit)” |

All parameters are taken directly from the specified table; no numerical values are listed here. The model above is ready for implementation in any standard linear‑programming solver.