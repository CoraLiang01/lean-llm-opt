**Mathematical Model**

Let  

* \(I=\{i\mid\text{discipline}_i=\text{“Operations Research”}\}\) – index set of OR courses.  
* \(x_i\in\{0,1\}\) – binary variable, 1 if course \(i\) is selected.  

Parameters (all taken from *courses_42.csv*)  

* \(c_i\) – credits of course \(i\).  
* \(p_i\) – interest‑points of course \(i\).  

Objective  

\[
\max \sum_{i\in I} p_i\,x_i
\]

Constraint  

\[
\sum_{i\in I} c_i\,x_i \;\le\; 20
\]

Variable domain  

\[
x_i\in\{0,1\}\qquad \forall i\in I
\]

---

**Data Mapping**

| Source file | Column name | Parameter / Index | Mapping rule |
|-------------|-------------|------------------|--------------|
| `Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv` | `discipline` | Index set \(I\) | Include row if `discipline` = `"Operations Research"` |
| `Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv` | `credits` | \(c_i\) | For each \(i\in I\), \(c_i =\) value in `credits` column |
| `Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv` | `interest_points` | \(p_i\) | For each \(i\in I\), \(p_i =\) value in `interest_points` column |

No other data or constraints are required. The model is a binary integer linear program that selects a subset of Operations Research courses with total credits not exceeding 20 to maximize total interest points.