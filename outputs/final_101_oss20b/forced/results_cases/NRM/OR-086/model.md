**Mathematical Model**

Let  

* \(G=\{I,II,III\}\) – raw‑grade indices  
* \(B=\{Red,Yellow,Blue\}\) – wine‑brand indices  

**Parameters**  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_g\) | unit cost of raw grade \(g\) (CNY/kg) | `file_0_view_0` – column **Cost (CNY/kg)** |
| \(s_g\) | daily supply limit of raw grade \(g\) (kg) | `file_0_view_0` – column **Daily Supply (kg)** |
| \(p_b\) | selling price of brand \(b\) (CNY/kg) | `file_1_view_0` – column **Selling Price (CNY/kg)** |
| \(L_{b,g}\) | lower‑bound proportion of grade \(g\) in brand \(b\) (fraction, 0–1) | derived from `file_1_view_0` – column **Blending Requirements** |
| \(U_{b,g}\) | upper‑bound proportion of grade \(g\) in brand \(b\) (fraction, 0–1) | derived from `file_1_view_0` – column **Blending Requirements** |

(If a particular grade has no lower or upper bound for a brand, the corresponding parameter is set to 0 or 1 so that the inequality is inactive.)

**Decision Variables**

* \(x_{g,b}\) – quantity of raw grade \(g\) used in brand \(b\) (kg)  \(x_{g,b}\ge 0\)  
* \(T_b\) – total production of brand \(b\) (kg)  \(T_b\ge 0\)

**Objective**

\[
\max \;\; \sum_{b\in B} p_b\,T_b \;-\; \sum_{g\in G}\sum_{b\in B} c_g\,x_{g,b}
\]

**Constraints**

1. **Production definition**  
   \[
   T_b = \sum_{g\in G} x_{g,b}\qquad \forall\,b\in B
   \]

2. **Blending proportions**  
   \[
   L_{b,g}\,T_b \;\le\; x_{g,b} \;\le\; U_{b,g}\,T_b
   \qquad \forall\,b\in B,\; g\in G
   \]

3. **Raw‑material supply limits**  
   \[
   \sum_{b\in B} x_{g,b} \;\le\; s_g \qquad \forall\,g\in G
   \]

4. **Minimum production for Red**  
   \[
   T_{\text{Red}} \;\ge\; 2000
   \]

All variables are continuous and non‑negative.

---

### Data Mapping

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(c_g\) | `file_0_view_0` | **Cost (CNY/kg)** |
| \(s_g\) | `file_0_view_0` | **Daily Supply (kg)** |
| \(p_b\) | `file_1_view_0` | **Selling Price (CNY/kg)** |
| \(L_{b,g}\), \(U_{b,g}\) | `file_1_view_0` | **Blending Requirements** (parsed into numeric fractions) |

The blending‑requirements column contains textual statements such as “I less than 10%  II more than 50%”. These are parsed to assign the corresponding lower/upper bounds \(L_{b,g}\) and \(U_{b,g}\) for each brand–grade pair. If a bound is not specified, the parameter is set to 0 (for lower) or 1 (for upper) so that the inequality is effectively inactive.