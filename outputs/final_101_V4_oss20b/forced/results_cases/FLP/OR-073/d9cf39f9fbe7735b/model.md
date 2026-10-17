**Symbolic Model**

Let  

* \(P=\{I,II,III\}\) products  
* \(E=\{A1,A2,A3,B1,B2,B3,B4\}\) equipment types  
* \(A=\{A1,A2,A3\}\) equipment that can perform procedure A  
* \(B=\{B1,B2,B3,B4\}\) equipment that can perform procedure B  

Decision variables  

\[
\begin{aligned}
x_{p,e}&\ge 0 &&\text{units of product }p\text{ processed on equipment }e,\\
y_{p}&\ge 0 &&\text{total units produced of product }p .
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
t_{p,e}&\ge 0 &&\text{processing time per unit of }p\text{ on }e,\\
T_e&\ge 0 &&\text{available operating time of equipment }e,\\
c_e&\ge 0 &&\text{equipment cost per unit time of }e,\\
r_p&\ge 0 &&\text{raw‑material cost per unit of }p,\\
p_p&\ge 0 &&\text{selling price per unit of }p,\\
\alpha_{p,e}&\in\{0,1\} &&\text{compatibility indicator (1 if }t_{p,e}\text{ is defined)} .
\end{aligned}
\]

Let \(M=\sum_{e\in E}T_e\) (a sufficiently large constant).

---

**Objective**

\[
\max \;\; \sum_{p\in P}\bigl(p_p-r_p\bigr)\,y_p
          \;-\;\sum_{e\in E}\sum_{p\in P} c_e\,t_{p,e}\,x_{p,e}
\]

---

**Constraints**

1. **Production consistency (procedure A)**  
   \[
   \sum_{e\in A} x_{p,e}=y_p \qquad \forall\,p\in P
   \]

2. **Production consistency (procedure B)**  
   \[
   \sum_{e\in B} x_{p,e}=y_p \qquad \forall\,p\in P
   \]

3. **Equipment time limits**  
   \[
   \sum_{p\in P} t_{p,e}\,x_{p,e}\;\le\;T_e \qquad \forall\,e\in E
   \]

4. **Compatibility (enforce zero for disallowed pairs)**  
   \[
   x_{p,e}\;\le\;M\,\alpha_{p,e}\qquad \forall\,p\in P,\;e\in E
   \]

5. **Non‑negativity**  
   \[
   x_{p,e}\ge 0,\;y_p\ge 0 \qquad \forall\,p\in P,\;e\in E
   \]

---

**Data Mapping (table `file_0_view_0`)**

| Parameter | Source column | Source row (equipment) | Notes |
|-----------|---------------|------------------------|-------|
| \(t_{p,e}\) | column *Product I*, *Product II*, *Product III* | row where *Equipment / Cost* = \(e\) | Blank → \(\alpha_{p,e}=0\) |
| \(T_e\) | column *Available Equipment Operating Time* | row where *Equipment / Cost* = \(e\) | |
| \(c_e\) | column *Equipment Cost at Full Load (yuan)* | row where *Equipment / Cost* = \(e\) | |
| \(r_p\) | column *Raw Material Cost (yuan/unit)* | row where *Equipment / Cost* = *Raw Material Cost (yuan/unit)* | |
| \(p_p\) | column *Unit Price (yuan/unit)* | row where *Equipment / Cost* = *Unit Price (yuan/unit)* | |
| \(\alpha_{p,e}\) | presence of a non‑blank entry in \(t_{p,e}\) | same as \(t_{p,e}\) | 1 if defined, 0 otherwise |

All other constants (e.g., \(M\)) are derived from the mapped data. No numerical values are enumerated in the model.