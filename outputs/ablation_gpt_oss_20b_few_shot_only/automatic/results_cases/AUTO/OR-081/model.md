**Mathematical model**

Let  

- \(F\) – set of all foods listed in *cost.csv*  
- For each \(f\in F\)  
  - \(c_f\) – calories per serving  
  - \(p_f\) – protein (g) per serving  
  - \(f_f\) – fat (g) per serving  
  - \(v_f\) – vitamin C (mg) per serving  
  - \(k_f\) – cost (USD) per serving  

Decision variable  

- \(x_f\ge 0\) – number of servings of food \(f\) (continuous, fractional servings allowed)

\[
\begin{aligned}
\min\;&\sum_{f\in F} k_f\,x_f \\[4pt]
\text{s.t.}\;&\sum_{f\in F} c_f\,x_f \;\ge\; 2000 \\[2pt]
&\sum_{f\in F} p_f\,x_f \;\ge\; 50 \\[2pt]
&\sum_{f\in F} v_f\,x_f \;\ge\; 60 \\[2pt]
&\sum_{f\in F} f_f\,x_f \;\le\; 70 \\[2pt]
&x_f \;\ge\; 0 \qquad \forall f\in F
\end{aligned}
\]

---

**Data‑mapping rules**

| Index set | Source file | Key column | Parameter columns |
|-----------|-------------|------------|-------------------|
| \(F\) | `Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv` | `Food` | `Calories`, `Protein(g)`, `Fat(g)`, `VitaminC(mg)`, `Cost` |

- Each row of the CSV defines one element \(f\in F\).  
- The values in the columns `Calories`, `Protein(g)`, `Fat(g)`, `VitaminC(mg)`, `Cost` are mapped directly to the parameters \(c_f\), \(p_f\), \(f_f\), \(v_f\), \(k_f\) respectively.  
- No additional filtering or aggregation is applied; all foods in the file are included in the decision set.  

The model above is a linear program with continuous decision variables and linear constraints, fully specified by the data mapping above.