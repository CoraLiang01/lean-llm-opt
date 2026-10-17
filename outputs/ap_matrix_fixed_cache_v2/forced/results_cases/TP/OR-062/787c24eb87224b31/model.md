##### Sets

Let  
$F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$ (suppliers)  
$S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$ (stores/customers)  
$C = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$ (demand records, mapped to stores as below)

##### Parameters

- Demands (from demand.csv):  
  $\text{demand}_{\text{CLARINDA}} = 2397$  
  $\text{demand}_{\text{FORT MADISON}} = 1889$  
  $\text{demand}_{\text{SIOUX CITY}} = 2518$  
  $\text{demand}_{\text{TOLEDO}} = 3218$  
  $\text{demand}_{\text{BANCROFT}} = 1813$  

- Fixed costs (from fixed_cost.csv):  
  $\text{fixed\_cost}_{\text{MOUNT AYR}} = 96.58$  
  $\text{fixed\_cost}_{\text{WAUKEE}} = 94.06$  
  $\text{fixed\_cost}_{\text{WAVERLY}} = 94.37$  
  $\text{fixed\_cost}_{\text{PELLA}} = 82.88$  
  $\text{fixed\_cost}_{\text{DES MOINES}} = 94.96$  

- Transportation costs (from transportation_costs.csv):  
  $\text{cost}_{i,j}$ for $i \in F$, $j \in S$:

| Supplier      | CLARINDA | FORT MADISON | SIOUX CITY | TOLEDO | BANCROFT |
|---------------|----------|--------------|------------|--------|----------|
| MOUNT AYR     | 694.68   | 17.48        | 20.07      | 199.02 | 1685.53  |
| WAUKEE        | 15.13    | 1.5          | 1.43       | 27.88  | 90.69    |
| WAVERLY       | 2.34     | 349.34       | 246.6      | 41.3   | 78.73    |
| PELLA         | 1181.6   | 1458.53      | 1646.36    | 1924.55| 38.93    |
| DES MOINES    | 1030.8   | 43.48        | 932.43     | 55.39  | 103.84   |

##### Decision Variables

- $y_i \in \{0,1\}$: $1$ if supplier $i$ is open, $0$ otherwise, for $i \in F$
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to store $j$, for $i \in F$, $j \in S$

##### Objective

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in F} \text{fixed\_cost}_i\, y_i + \sum_{i \in F} \sum_{j \in S} \text{cost}_{i,j}\, x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in S$,
   $$
   \sum_{i \in F} x_{ij} \geq \text{demand}_j
   $$

2. **Supplier activation:**  
   For all $i \in F$, $j \in S$,
   $$
   x_{ij} \leq M_{ij} y_i
   $$
   where $M_{ij}$ is a sufficiently large constant (e.g., $M_{ij} = \text{demand}_j$).

3. **Variable domains:**  
   $$
   y_i \in \{0,1\} \quad \forall i \in F
   $$
   $$
   x_{ij} \geq 0 \quad \forall i \in F,\, j \in S
   $$

##### Data

- $F = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$
- $S = \{\text{CLARINDA},\ \text{FORT MADISON},\ \text{SIOUX CITY},\ \text{TOLEDO},\ \text{BANCROFT}\}$
- Demands:
  - $\text{demand}_{\text{CLARINDA}} = 2397$
  - $\text{demand}_{\text{FORT MADISON}} = 1889$
  - $\text{demand}_{\text{SIOUX CITY}} = 2518$
  - $\text{demand}_{\text{TOLEDO}} = 3218$
  - $\text{demand}_{\text{BANCROFT}} = 1813$
- Fixed costs:
  - $\text{fixed\_cost}_{\text{MOUNT AYR}} = 96.58$
  - $\text{fixed\_cost}_{\text{WAUKEE}} = 94.06$
  - $\text{fixed\_cost}_{\text{WAVERLY}} = 94.37$
  - $\text{fixed\_cost}_{\text{PELLA}} = 82.88$
  - $\text{fixed\_cost}_{\text{DES MOINES}} = 94.96$
- Transportation costs:
  - See table above for $\text{cost}_{i,j}$

##### Complete Model

Minimize
$$
96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}}
$$
$$
+\,694.68\,x_{\text{MOUNT AYR},\text{CLARINDA}} + 17.48\,x_{\text{MOUNT AYR},\text{FORT MADISON}} + 20.07\,x_{\text{MOUNT AYR},\text{SIOUX CITY}} + 199.02\,x_{\text{MOUNT AYR},\text{TOLEDO}} + 1685.53\,x_{\text{MOUNT AYR},\text{BANCROFT}}
$$
$$
+\,15.13\,x_{\text{WAUKEE},\text{CLARINDA}} + 1.5\,x_{\text{WAUKEE},\text{FORT MADISON}} + 1.43\,x_{\text{WAUKEE},\text{SIOUX CITY}} + 27.88\,x_{\text{WAUKEE},\text{TOLEDO}} + 90.69\,x_{\text{WAUKEE},\text{BANCROFT}}
$$
$$
+\,2.34\,x_{\text{WAVERLY},\text{CLARINDA}} + 349.34\,x_{\text{WAVERLY},\text{FORT MADISON}} + 246.6\,x_{\text{WAVERLY},\text{SIOUX CITY}} + 41.3\,x_{\text{WAVERLY},\text{TOLEDO}} + 78.73\,x_{\text{WAVERLY},\text{BANCROFT}}
$$
$$
+\,1181.6\,x_{\text{PELLA},\text{CLARINDA}} + 1458.53\,x_{\text{PELLA},\text{FORT MADISON}} + 1646.36\,x_{\text{PELLA},\text{SIOUX CITY}} + 1924.55\,x_{\text{PELLA},\text{TOLEDO}} + 38.93\,x_{\text{PELLA},\text{BANCROFT}}
$$
$$
+\,1030.8\,x_{\text{DES MOINES},\text{CLARINDA}} + 43.48\,x_{\text{DES MOINES},\text{FORT MADISON}} + 932.43\,x_{\text{DES MOINES},\text{SIOUX CITY}} + 55.39\,x_{\text{DES MOINES},\text{TOLEDO}} + 103.84\,x_{\text{DES MOINES},\text{BANCROFT}}
$$

Subject to, for each store:
- $\sum_{i \in F} x_{i,\text{CLARINDA}} \geq 2397$
- $\sum_{i \in F} x_{i,\text{FORT MADISON}} \geq 1889$
- $\sum_{i \in F} x_{i,\text{SIOUX CITY}} \geq 2518$
- $\sum_{i \in F} x_{i,\text{TOLEDO}} \geq 3218$
- $\sum_{i \in F} x_{i,\text{BANCROFT}} \geq 1813$

For all $i \in F$, $j \in S$:
- $x_{ij} \leq \text{demand}_j\, y_i$
- $x_{ij} \geq 0$

For all $i \in F$:
- $y_i \in \{0,1\}$

All identifiers and coefficients are preserved as in the retrieved data.