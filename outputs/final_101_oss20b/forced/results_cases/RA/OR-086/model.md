**Mathematical Model**

Let  

* \(G=\{I,II,III\}\) – set of raw‑wine grades  
* \(B=\{\text{Red},\text{Yellow},\text{Blue}\}\) – set of wine brands  

Decision variables  
\[
x_{g,b}\;\;(\text{kg})\qquad g\in G,\;b\in B
\]

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(S_g\) | Daily supply limit of grade \(g\) (kg) | `file_0_view_0` – column **Daily Supply (kg)** |
| \(c_g\) | Unit cost of grade \(g\) (CNY/kg) | `file_0_view_0` – column **Cost (CNY/kg)** |
| \(p_b\) | Selling price of brand \(b\) (CNY/kg) | `file_1_view_0` – column **Selling Price (CNY/kg)** |
| \(\ell_{g,b}\) | Lower proportion bound of grade \(g\) in brand \(b\) (fraction) | Derived from `file_1_view_0` – **Blending Requirements** |
| \(u_{g,b}\) | Upper proportion bound of grade \(g\) in brand \(b\) (fraction) | Derived from `file_1_view_0` – **Blending Requirements** |

(If a brand has no explicit bound for a grade, \(\ell_{g,b}=0\) and \(u_{g,b}=1\).)

---

### Objective
\[
\max \;\; \sum_{b\in B} p_b \Bigl(\sum_{g\in G} x_{g,b}\Bigr)
          \;-\; \sum_{g\in G} c_g \Bigl(\sum_{b\in B} x_{g,b}\Bigr)
\]

### Constraints

1. **Blending proportion bounds**  
   For every brand \(b\) and grade \(g\):
   \[
   \ell_{g,b}\;\Bigl(\sum_{h\in G} x_{h,b}\Bigr) \;\le\; x_{g,b}
   \;\le\; u_{g,b}\;\Bigl(\sum_{h\in G} x_{h,b}\Bigr)
   \]

2. **Raw‑material supply limits**  
   For every grade \(g\):
   \[
   \sum_{b\in B} x_{g,b} \;\le\; S_g
   \]

3. **Minimum production for Red**  
   \[
   \sum_{g\in G} x_{g,\text{Red}} \;\ge\; 2000
   \]

4. **Non‑negativity**  
   \[
   x_{g,b} \;\ge\; 0 \qquad \forall g\in G,\;b\in B
   \]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(S_g\) | `file_0_view_0` | `Daily Supply (kg)` |
| \(c_g\) | `file_0_view_0` | `Cost (CNY/kg)` |
| \(p_b\) | `file_1_view_0` | `Selling Price (CNY/kg)` |
| \(\ell_{g,b}\), \(u_{g,b}\) | `file_1_view_0` | `Blending Requirements` (parsed to extract bounds) |

*The blending bounds \(\ell_{g,b}\) and \(u_{g,b}\) are obtained by parsing the textual “Blending Requirements” field for each brand:  
- “\(g\) less than \(X\%\)” → \(u_{g,b}=X/100\).  
- “\(g\) more than \(Y\%\)” → \(\ell_{g,b}=Y/100\).  
If a grade is not mentioned, its bounds default to 0 and 1.*