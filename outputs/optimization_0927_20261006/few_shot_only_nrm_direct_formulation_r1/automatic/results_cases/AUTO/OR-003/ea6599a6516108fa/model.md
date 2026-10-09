**Sets:**

- Suppliers $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- Customers $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

**Parameters:**

- Customer demands $d_j$:

  - $d_{\text{C1}} = 216$
  - $d_{\text{C2}} = 168$
  - $d_{\text{C3}} = 264$
  - $d_{\text{C4}} = 216$
  - $d_{\text{C5}} = 216$
  - $d_{\text{C6}} = 192$
  - $d_{\text{C7}} = 144$
  - $d_{\text{C8}} = 168$
  - $d_{\text{C9}} = 168$
  - $d_{\text{C10}} = 168$

- Supplier capacities $s_i$:

  - $s_{\text{S1}} = 288$
  - $s_{\text{S2}} = 288$
  - $s_{\text{S3}} = 264$
  - $s_{\text{S4}} = 264$
  - $s_{\text{S5}} = 216$
  - $s_{\text{S6}} = 216$
  - $s_{\text{S7}} = 168$
  - $s_{\text{S8}} = 216$
  - $s_{\text{S9}} = 240$
  - $s_{\text{S10}} = 168$

- Transportation costs $c_{ij}$ (per unit from supplier $i$ to customer $j$):

|        |   C1         |   C2         |   C3         |   C4         |   C5         |   C6         |   C7         |   C8         |   C9         |   C10        |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1     | 590.36481365 | 23.66917261  | 88.89005870  | 497.52228807 | 466.09034322 | 29.02209683  | 23.67524483  | 23.67776029  | 0.31183949   | 58.89547392  |
| S2     | 2042.0715002 | 2133.9784843 | 705.15912034 | 101.59454516 | 2052.9376574 | 1738.7549514 | 101.61094966 | 101.61062174 | 122.45214269 | 67.29170751  |
| S3     | 22.29722216  | 497.92719393 | 1653.0828862 | 23.68545123  | 1386.0807887 | 26.13715281  | 497.62204829 | 498.09358471 | 865.38162963 | 1008.6717395 |
| S4     | 960.78145339 | 49.12830005  | 1324.2386971 | 1032.2095478 | 0.07804725   | 53.30826873  | 49.13641672  | 1031.8214425 | 466.00495308 | 1351.8189071 |
| S5     | 1471.2721666 | 85.69560728  | 38.89266824  | 1542.0500358 | 112.20514372 | 82.37020164  | 1542.3399197 | 85.69238746  | 1924.9360769 | 1094.6695961 |
| S6     | 191.90587261 | 158.50401032 | 91.02045350  | 184.44747202 | 968.14679870 | 284.10760621 | 8.79106159   | 158.70523836 | 27.94387435  | 929.80716828 |
| S7     | 81.23891457  | 0.37446422   | 2079.4668654 | 0.30656718   | 1031.7772962 | 7.20396449   | 0.07623072   | 0.03247388   | 23.68582797  | 849.97994066 |
| S8     | 56.09931097  | 935.61431087 | 73.08824617  | 52.00392409  | 4.02579239   | 1002.2327658 | 935.77660296 | 935.70072523 | 612.86987193 | 1348.8366146 |
| S9     | 4.50228333   | 0.38995853   | 1782.4662178 | 0.00634591   | 1031.9910115 | 129.50665620 | 0.21183196   | 0.64573011   | 497.62723911 | 40.46575555  |
| S10    | 333.68692704 | 277.47193861 | 86.02096892  | 277.30836609 | 1004.4649085 | 19.95033682  | 13.20207369  | 238.14321522 | 411.05803324 | 941.75263656 |

**Decision Variables:**

- $x_{ij} \geq 0$ (continuous): quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

**Objective:**

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

That is,

\[
\min \Bigg(
\begin{aligned}
&590.36481365\,x_{\text{S1},\text{C1}} + 23.66917261\,x_{\text{S1},\text{C2}} + 88.89005870\,x_{\text{S1},\text{C3}} + 497.52228807\,x_{\text{S1},\text{C4}} + 466.09034322\,x_{\text{S1},\text{C5}} \\
&+ 29.02209683\,x_{\text{S1},\text{C6}} + 23.67524483\,x_{\text{S1},\text{C7}} + 23.67776029\,x_{\text{S1},\text{C8}} + 0.31183949\,x_{\text{S1},\text{C9}} + 58.89547392\,x_{\text{S1},\text{C10}} \\
&+ 2042.0715002\,x_{\text{S2},\text{C1}} + 2133.9784843\,x_{\text{S2},\text{C2}} + 705.15912034\,x_{\text{S2},\text{C3}} + 101.59454516\,x_{\text{S2},\text{C4}} + 2052.9376574\,x_{\text{S2},\text{C5}} \\
&+ 1738.7549514\,x_{\text{S2},\text{C6}} + 101.61094966\,x_{\text{S2},\text{C7}} + 101.61062174\,x_{\text{S2},\text{C8}} + 122.45214269\,x_{\text{S2},\text{C9}} + 67.29170751\,x_{\text{S2},\text{C10}} \\
&+ 22.29722216\,x_{\text{S3},\text{C1}} + 497.92719393\,x_{\text{S3},\text{C2}} + 1653.0828862\,x_{\text{S3},\text{C3}} + 23.68545123\,x_{\text{S3},\text{C4}} + 1386.0807887\,x_{\text{S3},\text{C5}} \\
&+ 26.13715281\,x_{\text{S3},\text{C6}} + 497.62204829\,x_{\text{S3},\text{C7}} + 498.09358471\,x_{\text{S3},\text{C8}} + 865.38162963\,x_{\text{S3},\text{C9}} + 1008.6717395\,x_{\text{S3},\text{C10}} \\
&+ 960.78145339\,x_{\text{S4},\text{C1}} + 49.12830005\,x_{\text{S4},\text{C2}} + 1324.2386971\,x_{\text{S4},\text{C3}} + 1032.2095478\,x_{\text{S4},\text{C4}} + 0.07804725\,x_{\text{S4},\text{C5}} \\
&+ 53.30826873\,x_{\text{S4},\text{C6}} + 49.13641672\,x_{\text{S4},\text{C7}} + 1031.8214425\,x_{\text{S4},\text{C8}} + 466.00495308\,x_{\text{S4},\text{C9}} + 1351.8189071\,x_{\text{S4},\text{C10}} \\
&+ 1471.2721666\,x_{\text{S5},\text{C1}} + 85.69560728\,x_{\text{S5},\text{C2}} + 38.89266824\,x_{\text{S5},\text{C3}} + 1542.0500358\,x_{\text{S5},\text{C4}} + 112.20514372\,x_{\text{S5},\text{C5}} \\
&+ 82.37020164\,x_{\text{S5},\text{C6}} + 1542.3399197\,x_{\text{S5},\text{C7}} + 85.69238746\,x_{\text{S5},\text{C8}} + 1924.9360769\,x_{\text{S5},\text{C9}} + 1094.6695961\,x_{\text{S5},\text{C10}} \\
&+ 191.90587261\,x_{\text{S6},\text{C1}} + 158.50401032\,x_{\text{S6},\text{C2}} + 91.02045350\,x_{\text{S6},\text{C3}} + 184.44747202\,x_{\text{S6},\text{C4}} + 968.14679870\,x_{\text{S6},\text{C5}} \\
&+ 284.10760621\,x_{\text{S6},\text{C6}} + 8.79106159\,x_{\text{S6},\text{C7}} + 158.70523836\,x_{\text{S6},\text{C8}} + 27.94387435\,x_{\text{S6},\text{C9}} + 929.80716828\,x_{\text{S6},\text{C10}} \\
&+ 81.23891457\,x_{\text{S7},\text{C1}} + 0.37446422\,x_{\text{S7},\text{C2}} + 2079.4668654\,x_{\text{S7},\text{C3}} + 0.30656718\,x_{\text{S7},\text{C4}} + 1031.7772962\,x_{\text{S7},\text{C5}} \\
&+ 7.20396449\,x_{\text{S7},\text{C6}} + 0.07623072\,x_{\text{S7},\text{C7}} + 0.03247388\,x_{\text{S7},\text{C8}} + 23.68582797\,x_{\text{S7},\text{C9}} + 849.97994066\,x_{\text{S7},\text{C10}} \\
&+ 56.09931097\,x_{\text{S8},\text{C1}} + 935.61431087\,x_{\text{S8},\text{C2}} + 73.08824617\,x_{\text{S8},\text{C3}} + 52.00392409\,x_{\text{S8},\text{C4}} + 4.02579239\,x_{\text{S8},\text{C5}} \\
&+ 1002.2327658\,x_{\text{S8},\text{C6}} + 935.77660296\,x_{\text{S8},\text{C7}} + 935.70072523\,x_{\text{S8},\text{C8}} + 612.86987193\,x_{\text{S8},\text{C9}} + 1348.8366146\,x_{\text{S8},\text{C10}} \\
&+ 4.50228333\,x_{\text{S9},\text{C1}} + 0.38995853\,x_{\text{S9},\text{C2}} + 1782.4662178\,x_{\text{S9},\text{C3}} + 0.00634591\,x_{\text{S9},\text{C4}} + 1031.9910115\,x_{\text{S9},\text{C5}} \\
&+ 129.50665620\,x_{\text{S9},\text{C6}} + 0.21183196\,x_{\text{S9},\text{C7}} + 0.64573011\,x_{\text{S9},\text{C8}} + 497.62723911\,x_{\text{S9},\text{C9}} + 40.46575555\,x_{\text{S9},\text{C10}} \\
&+ 333.68692704\,x_{\text{S10},\text{C1}} + 277.47193861\,x_{\text{S10},\text{C2}} + 86.02096892\,x_{\text{S10},\text{C3}} + 277.30836609\,x_{\text{S10},\text{C4}} + 1004.4649085\,x_{\text{S10},\text{C5}} \\
&+ 19.95033682\,x_{\text{S10},\text{C6}} + 13.20207369\,x_{\text{S10},\text{C7}} + 238.14321522\,x_{\text{S10},\text{C8}} + 411.05803324\,x_{\text{S10},\text{C9}} + 941.75263656\,x_{\text{S10},\text{C10}}
\end{aligned}
\Bigg)
\]

**Subject to:**

1. **Demand satisfaction (for each customer $j$):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
   That is,
   \begin{align*}
   x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + \cdots + x_{\text{S10},\text{C1}} &\geq 216 \\
   x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + \cdots + x_{\text{S10},\text{C2}} &\geq 168 \\
   x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + \cdots + x_{\text{S10},\text{C3}} &\geq 264 \\
   x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + \cdots + x_{\text{S10},\text{C4}} &\geq 216 \\
   x_{\text{S1},\text{C5}} + x_{\text{S2},\text{C5}} + \cdots + x_{\text{S10},\text{C5}} &\geq 216 \\
   x_{\text{S1},\text{C6}} + x_{\text{S2},\text{C6}} + \cdots + x_{\text{S10},\text{C6}} &\geq 192 \\
   x_{\text{S1},\text{C7}} + x_{\text{S2},\text{C7}} + \cdots + x_{\text{S10},\text{C7}} &\geq 144 \\
   x_{\text{S1},\text{C8}} + x_{\text{S2},\text{C8}} + \cdots + x_{\text{S10},\text{C8}} &\geq 168 \\
   x_{\text{S1},\text{C9}} + x_{\text{S2},\text{C9}} + \cdots + x_{\text{S10},\text{C9}} &\geq 168 \\
   x_{\text{S1},\text{C10}} + x_{\text{S2},\text{C10}} + \cdots + x_{\text{S10},\text{C10}} &\geq 168 \\
   \end{align*}

2. **Supply capacity (for each supplier $i$):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
   That is,
   \begin{align*}
   x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + \cdots + x_{\text{S1},\text{C10}} &\leq 288 \\
   x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + \cdots + x_{\text{S2},\text{C10}} &\leq 288 \\
   x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + \cdots + x_{\text{S3},\text{C10}} &\leq 264 \\
   x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + \cdots + x_{\text{S4},\text{C10}} &\leq 264 \\
   x_{\text{S5},\text{C1}} + x_{\text{S5},\text{C2}} + \cdots + x_{\text{S5},\text{C10}} &\leq 216 \\
   x_{\text{S6},\text{C1}} + x_{\text{S6},\text{C2}} + \cdots + x_{\text{S6},\text{C10}} &\leq 216 \\
   x_{\text{S7},\text{C1}} + x_{\text{S7},\text{C2}} + \cdots + x_{\text{S7},\text{C10}} &\leq 168 \\
   x_{\text{S8},\text{C1}} + x_{\text{S8},\text{C2}} + \cdots + x_{\text{S8},\text{C10}} &\leq 216 \\
   x_{\text{S9},\text{C1}} + x_{\text{S9},\text{C2}} + \cdots + x_{\text{S9},\text{C10}} &\leq 240 \\
   x_{\text{S10},\text{C1}} + x_{\text{S10},\text{C2}} + \cdots + x_{\text{S10},\text{C10}} &\leq 168 \\
   \end{align*}

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

**All identifiers and coefficients are preserved exactly as retrieved.**