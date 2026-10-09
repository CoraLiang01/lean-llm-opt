##### Abstract Mathematical Model

Let  
- $I$ = set of drug types (indexed by $i$), from products.csv ProductName  
- $b_i$ = benefit coefficient of drug $i$, from products.csv Value  
- $w_i$ = weight per unit of drug $i$, from products.csv Weight  
- $C$ = total inventory capacity, from capacity.csv Capacity  
- $x_i$ = number of units of drug $i$ to order daily (decision variable), integer and $\geq 0$

**Objective:**  
$\max \sum_{i \in I} b_i x_i$

**Subject to:**  
$\sum_{i \in I} w_i x_i \leq C$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

##### Data Mapping

- $I$: All records in products.csv, column ProductName
- $b_i$: products.csv, column Value, keyed by ProductName
- $w_i$: products.csv, column Weight, keyed by ProductName
- $C$: capacity.csv, column Capacity
- $x_i$: decision variable for each $i \in I$ (drug type)