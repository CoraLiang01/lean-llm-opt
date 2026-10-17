**Mathematical Model**

Let  

* \(I=\{1,2,\dots ,110\}\) – set of all projects.  
* \(x_i\in\{0,1\}\) – binary variable, 1 if project \(i\) is selected, 0 otherwise.  

Parameters (all taken from *project.csv*, table id **file_0_view_0**):

| Parameter | Definition | Source column |
|-----------|------------|---------------|
| \(c_i\) | Capital requirement of project \(i\) (k$) | “Capital (k$)” |
| \(p_i\) | Expected NPV of project \(i\) (k$) | “NPV (k$)” |

---

### Objective
\[
\max \;\; \sum_{i\in I} p_i\,x_i
\]

### Constraints

1. **Budget limit**  
   \[
   \sum_{i\in I} c_i\,x_i \;\le\; 1000
   \]

2. **Mutually exclusive (Projects 4 & 7)**  
   \[
   x_4 + x_7 \;\le\; 1
   \]

3. **Prerequisite (Project 6 requires 1)**  
   \[
   x_6 \;\le\; x_1
   \]

4. **Contingent (Project 10 requires 5)**  
   \[
   x_{10} \;\le\; x_5
   \]

5. **Binary decision variables**  
   \[
   x_i \in \{0,1\}\quad \forall i\in I
   \]

---

**Data Mapping**

| Table ID | Column | Parameter |
|----------|--------|-----------|
| file_0_view_0 | “Capital (k$)” | \(c_i\) |
| file_0_view_0 | “NPV (k$)” | \(p_i\) |

All indices \(i\) in the model correspond to the rows of *project.csv* (Project ID). No other data are required.