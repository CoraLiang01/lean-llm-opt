Let $x_{ij}$ denote the quantity shipped from warehouse (region) $i$ to store (customer) $j$.

#### Objective:
Minimize total transportation cost:
$$
\min \sum_{i \in \{\text{S1},\text{S2},\text{S3},\text{S4},\text{S5}\}} \sum_{j \in \{\text{D1},\text{D2},\text{D3},\text{D4},\text{D5}\}} c_{ij} x_{ij}
$$
where $c_{ij}$ is the unit transportation cost from warehouse $i$ to store $j$ as given below.

#### Parameters:

- Warehouses (regions): S1, S2, S3, S4, S5
- Stores (customers): D1, D2, D3, D4, D5

- Supply capacities:
  - S1: 428
  - S2: 217
  - S3: 214
  - S4: 380
  - S5: 254

- Store demands:
  - D1: 428
  - D2: 217
  - D3: 214
  - D4: 380
  - D5: 254

- Transportation costs $c_{ij}$:

|        | D1              | D2              | D3              | D4              | D5              |
|--------|-----------------|-----------------|-----------------|-----------------|-----------------|
| S1     | 269.39105880208 | 1.45373353909   | 99.60345345757  | 26.64078166310  | 9.53768895688   |
| S2     | 9.29184687679   | 10.87477843707  | 144.52609291615 | 11.42013307790  | 153.17568199278 |
| S3     | 9.67458430167   | 2.61916509597   | 100.82422491687 | 3.21219108879   | 133.84933961242 |
| S4     | 270.57498480010 | 32.50253586     | 4.68420980965   | 1.56822696865   | 9.58927599      |
| S5     | 226.03319106758 | 8.66916198083   | 65.47681316968  | 9.06876525846   | 202.65015316426 |

#### Constraints:

1. **Supply capacity at each warehouse:**
   $$
   \sum_{j \in \{\text{D1},\text{D2},\text{D3},\text{D4},\text{D5}\}} x_{ij} \leq \text{supply\_capacity}_i, \quad \forall i \in \{\text{S1},\text{S2},\text{S3},\text{S4},\text{S5}\}
   $$
   - S1: $\sum_j x_{\text{S1},j} \leq 428$
   - S2: $\sum_j x_{\text{S2},j} \leq 217$
   - S3: $\sum_j x_{\text{S3},j} \leq 214$
   - S4: $\sum_j x_{\text{S4},j} \leq 380$
   - S5: $\sum_j x_{\text{S5},j} \leq 254$

2. **Demand satisfaction at each store:**
   $$
   \sum_{i \in \{\text{S1},\text{S2},\text{S3},\text{S4},\text{S5}\}} x_{ij} = \text{demand}_j, \quad \forall j \in \{\text{D1},\text{D2},\text{D3},\text{D4},\text{D5}\}
   $$
   - D1: $\sum_i x_{i,\text{D1}} = 428$
   - D2: $\sum_i x_{i,\text{D2}} = 217$
   - D3: $\sum_i x_{i,\text{D3}} = 214$
   - D4: $\sum_i x_{i,\text{D4}} = 380$
   - D5: $\sum_i x_{i,\text{D5}} = 254$

3. **Nonnegativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in \{\text{S1},\text{S2},\text{S3},\text{S4},\text{S5}\},\ j \in \{\text{D1},\text{D2},\text{D3},\text{D4},\text{D5}\}
   $$

#### Complete Model (with all coefficients):

Minimize
$$
269.39105880208\,x_{\text{S1},\text{D1}} + 1.45373353909\,x_{\text{S1},\text{D2}} + 99.60345345757\,x_{\text{S1},\text{D3}} + 26.64078166310\,x_{\text{S1},\text{D4}} + 9.53768895688\,x_{\text{S1},\text{D5}} \\
+ 9.29184687679\,x_{\text{S2},\text{D1}} + 10.87477843707\,x_{\text{S2},\text{D2}} + 144.52609291615\,x_{\text{S2},\text{D3}} + 11.42013307790\,x_{\text{S2},\text{D4}} + 153.17568199278\,x_{\text{S2},\text{D5}} \\
+ 9.67458430167\,x_{\text{S3},\text{D1}} + 2.61916509597\,x_{\text{S3},\text{D2}} + 100.82422491687\,x_{\text{S3},\text{D3}} + 3.21219108879\,x_{\text{S3},\text{D4}} + 133.84933961242\,x_{\text{S3},\text{D5}} \\
+ 270.57498480010\,x_{\text{S4},\text{D1}} + 32.50253586\,x_{\text{S4},\text{D2}} + 4.68420980965\,x_{\text{S4},\text{D3}} + 1.56822696865\,x_{\text{S4},\text{D4}} + 9.58927599\,x_{\text{S4},\text{D5}} \\
+ 226.03319106758\,x_{\text{S5},\text{D1}} + 8.66916198083\,x_{\text{S5},\text{D2}} + 65.47681316968\,x_{\text{S5},\text{D3}} + 9.06876525846\,x_{\text{S5},\text{D4}} + 202.65015316426\,x_{\text{S5},\text{D5}}
$$

Subject to:
- $\sum_{j} x_{\text{S1},j} \leq 428$
- $\sum_{j} x_{\text{S2},j} \leq 217$
- $\sum_{j} x_{\text{S3},j} \leq 214$
- $\sum_{j} x_{\text{S4},j} \leq 380$
- $\sum_{j} x_{\text{S5},j} \leq 254$

- $\sum_{i} x_{i,\text{D1}} = 428$
- $\sum_{i} x_{i,\text{D2}} = 217$
- $\sum_{i} x_{i,\text{D3}} = 214$
- $\sum_{i} x_{i,\text{D4}} = 380$
- $\sum_{i} x_{i,\text{D5}} = 254$

- $x_{ij} \geq 0$ for all $i,j$ as above.