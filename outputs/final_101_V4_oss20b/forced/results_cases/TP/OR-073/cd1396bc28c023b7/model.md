**Mathematical model**

Let  

* \(P=\{I,II,III\}\) – product types  
* \(A=\{A1,A2,A3\}\) – equipment that performs procedure A  
* \(B=\{B1,B2,B3,B4\}\) – equipment that performs procedure B  

Parameters (all non‑negative)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(t_{p,e}^{A}\) | processing time per unit of product \(p\) on equipment \(e\in A\) | table `file_0_view_0`, columns “Product I/II/III” for rows with `Equipment / Cost` in \(A\) |
| \(t_{p,e}^{B}\) | processing time per unit of product \(p\) on equipment \(e\in B\) | table `file_0_view_0`, columns “Product I/II/III” for rows with `Equipment / Cost` in \(B\) |
| \(a_e\) | available operating time of equipment \(e\) | table `file_0_view_0`, column “Available Equipment Operating Time” |
| \(c_e\) | equipment cost per hour of equipment \(e\) | table `file_0_view_0`, column “Equipment Cost at Full Load (yuan)” |
| \(r_p\) | raw‑material cost per unit of product \(p\) | table `file_0_view_0`, row “Raw Material Cost (yuan/unit)” |
| \(p_p\) | unit selling price of product \(p\) | table `file_0_view_0`, row “Unit Price (yuan/unit)” |

Decision variables  

* \(x_{p,e}^{A}\ge 0\) – quantity of product \(p\) processed on equipment \(e\in A\) (procedure A)  
* \(x_{p,e}^{B}\ge 0\) – quantity of product \(p\) processed on equipment \(e\in B\) (procedure B)  
* \(y_p\ge 0\) – total production quantity of product \(p\)

**Objective**

\[
\max \;\; \sum_{p\in P} p_p\,y_p \;-\; \sum_{p\in P} r_p\,y_p
\;-\; \sum_{e\in A}\; c_e \sum_{p\in P} t_{p,e}^{A}\,x_{p,e}^{A}
\;-\; \sum_{e\in B}\; c_e \sum_{p\in P} t_{p,e}^{B}\,x_{p,e}^{B}
\]

**Constraints**

1. **Production consistency** – each unit of a product must be processed on both procedures  

\[
\sum_{e\in A} x_{p,e}^{A} \;=\; \sum_{e\in B} x_{p,e}^{B} \;=\; y_p ,
\qquad \forall\,p\in P
\]

2. **Equipment availability** – total time used on each equipment cannot exceed its available time  

\[
\sum_{p\in P} t_{p,e}^{A}\,x_{p,e}^{A} \;\le\; a_e ,
\qquad \forall\,e\in A
\]

\[
\sum_{p\in P} t_{p,e}^{B}\,x_{p,e}^{B} \;\le\; a_e ,
\qquad \forall\,e\in B
\]

3. **Feasibility of processing** – variables are zero when the corresponding processing time is undefined (empty cell in the data)  

\[
x_{p,e}^{A}=0 \;\text{if}\; t_{p,e}^{A}\text{ is missing},\qquad
x_{p,e}^{B}=0 \;\text{if}\; t_{p,e}^{B}\text{ is missing}
\]

4. **Non‑negativity** – already stated in variable definitions.

---

### Data‑mapping rules

| Table ID | Source row | Mapping |
|----------|------------|---------|
| `file_0_view_0` | 0–6 | For each row: <br>• `Equipment / Cost` → equipment id \(e\) <br>• `Available Equipment Operating Time` → \(a_e\) <br>• `Equipment Cost at Full Load (yuan)` → \(c_e\) <br>• `Product I` → \(t_{I,e}^{A}\) if \(e\in A\) else \(t_{I,e}^{B}\) <br>• `Product II` → \(t_{II,e}^{A/B}\) <br>• `Product III` → \(t_{III,e}^{A/B}\) |
| `file_0_view_0` | 7 | `Raw Material Cost (yuan/unit)` → \(r_I,r_{II},r_{III}\) |
| `file_0_view_0` | 8 | `Unit Price (yuan/unit)` → \(p_I,p_{II},p_{III}\) |

All numeric values are taken exactly as listed in the CSV; missing entries are treated as “undefined” and enforce the feasibility constraints above. No additional data are introduced.