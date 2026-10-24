**

The model is a mixed‑integer linear program that maximises the net benefit of ordering air‑conditioner models.  
Decision variables  

* `q[i]` – integer quantity of item `i` (≥ 0).  
* `z[i]` – binary, 1 if item `i` is selected (i.e., `q[i] > 0`).  
* `w[c]` – binary, 1 if category `c` is used.  
* `y[k]` – binary, 1 if both items of bundle `k` are selected.  

Objective  

\[
\max \sum_{i} a_i\,q_i \;-\; \sum_{i} f_i\,z_i \;-\; \sum_{c} F_c\,w_c \;+\; \sum_{k} \text{bonus}_k\,y_k
\]

Subject to the constraints described in the problem statement (item bounds, category limits, resource capacity, incompatibilities, prerequisites, bundle activation, and variable domains).  

---  

**CODE:**