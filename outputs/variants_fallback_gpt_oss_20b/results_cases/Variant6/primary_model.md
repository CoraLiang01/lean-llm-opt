**Model**

Let  

* \(F\) – set of product families (obtained from *option_catalog.csv*).  
* \(O_f\) – set of options belonging to family \(f\).  

Decision variables  

\[
x_{f,o}\in\{0,1\}\qquad \forall f\in F,\; o\in O_f
\]

Parameters  

\[
v_{f,o}\;(\text{value}),\; w_{f,o}\;(\text{weight}),\; b_{f,o}\;(\text{budget use})
\]
from *option_catalog.csv*;  

\[
W_{\max}\;(55),\; B_{\max}\;(70)
\]
from *resource_limits.csv*.

**Objective**

\[
\max \;\sum_{f\in F}\sum_{o\in O_f} v_{f,o}\,x_{f,o}
\]

**Constraints**

1. **One option per family**

\[
\sum_{o\in O_f} x_{f,o}=1 \qquad \forall f\in F
\]

2. **Weight limit**

\[
\sum_{f\in F}\sum_{o\in O_f} w_{f,o}\,x_{f,o}\;\le\; W_{\max}
\]

3. **Budget limit**

\[
\sum_{f\in F}\sum_{o\in O_f} b_{f,o}\,x_{f,o}\;\le\; B_{\max}
\]

4. **Binary variables**

\[
x_{f,o}\in\{0,1\}\qquad \forall f,o
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(v_{f,o}\) | `file_0_view_0` | `Value` | value of option \(o\) in family \(f\) |
| \(w_{f,o}\) | `file_0_view_0` | `Weight` | weight of option \(o\) in family \(f\) |
| \(b_{f,o}\) | `file_0_view_0` | `BudgetUse` | budget use of option \(o\) in family \(f\) |
| \(W_{\max}\) | `file_1_view_0` | `Limit` where `Resource`=`Weight` | overall weight limit |
| \(B_{\max}\) | `file_1_view_0` | `Limit` where `Resource`=`BudgetUse` | overall budget limit |

The index sets \(F\) and \(O_f\) are taken from all distinct `Family` and `Option` values present in *option_catalog.csv*. No additional rows or columns are introduced beyond those explicitly listed in the data mapping.