##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity of goods shipped from distribution center (supplier) $i$ to customer group $j$.

Where:
- $i \in I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$
- $j \in J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$

##### Parameters

- $d_j$: demand (units) for customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity (units) for supplier $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from transportation_costs.csv)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction** (each customer group must receive at least its demand):

   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity** (each supplier cannot ship more than its capacity):

   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity**:

   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

---

#### Data

**Customer Demands** (from customer_demand.csv):

| Customer | Demand ($d_j$) |
|----------|---------------|
| C1  | 4415 |
| C2  | 5430 |
| C3  | 81   |
| C4  | 146  |
| C5  | 10638|
| C6  | 1663 |
| C7  | 151  |
| C8  | 185  |
| C9  | 1917 |
| C10 | 4489 |
| C11 | 76   |
| C12 | 2529 |
| C13 | 2136 |
| C14 | 909  |
| C15 | 316  |
| C16 | 70   |
| C17 | 1456 |
| C18 | 2711 |

**Supplier Capacities** (from supply_capacity.csv):

| Supplier | Capacity ($s_i$) |
|----------|-----------------|
| S1  | 5963  |
| S2  | 702   |
| S3  | 350   |
| S4  | 11483 |
| S5  | 6585  |
| S6  | 11330 |
| S7  | 207   |
| S8  | 788   |
| S9  | 6967  |
| S10 | 43    |
| S11 | 1137  |
| S12 | 1553  |
| S13 | 257   |
| S14 | 2114  |
| S15 | 205   |
| S16 | 17326 |
| S17 | 22260 |
| S18 | 333   |

**Transportation Costs** (from transportation_costs.csv):

Let $c_{ij}$ be the cost from supplier $i$ to customer $j$ as follows (partial table shown for clarity; all values are included in the data):

| Supplier | C1         | C2         | C3         | C4         | C5         | C6         | C7         | C8         | C9         | C10        | C11        | C12        | C13        | C14        | C15        | C16        | C17        | C18        |
|----------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| S1  | 159.83765495858208 | 6.42633790337285 | 7.582243029047409 | 99.52254454692138 | 135.16829918665007 | 7.590909409397181 | 2.3654667052579508 | 7.575274779934459 | 991.7798969083236 | 11.464155019446489 | 69.35418474260145 | 135.88274162390056 | 46.86156560390732 | 7.591284332304382 | 159.41034277497943 | 7.560918384716748 | 173.2178262613133 | 7.6626195540288755 |
| S2  | 0.106557017102239 | 8.54894494648944 | 0.0568789303102096 | 212.07495813173492 | 1.040260758093917 | 0.0652083182686696 | 194.7110872056446 | 0.0494806010515883 | 839.3507562340868 | 89.40158875493893 | 217.73477041745696 | 68.76890997932357 | 141.20103295662906 | 0.0656502397018673 | 1.175025378373934 | 1.265678945608808 | 132.53399287398733 | 0.1438122345864802 |
| S3  | 0.0712113296922013 | 180.3996497782258 | 0.2622972046955685 | 182.8636027839307 | 1.735658687947286 | 0.0003457967259966 | 9.336781626424909 | 0.2872050597506193 | 39.90767148745277 | 4.201696016865375 | 187.79672431983445 | 3.3050624901211525 | 142.30344029228047 | 0.0127896657555688 | 0.003164519819296 | 1.724132015627177 | 113.56966737977794 | 0.0721377516383126 |
| S4  | 1.5455382897983383 | 180.07836268028748 | 0.0240569354592471 | 182.8162278499576 | 0.096483806717389 | 0.3280567912491164 | 167.94349941079716 | 0.3447071113014559 | 838.3178615319802 | 75.88119317030038 | 218.9576990348591 | 3.289687537275121 | 165.76290434538876 | 0.0150577528830554 | 0.0155049780044053 | 1.5358539138142018 | 132.22868668242745 | 0.0807047191286277 |
| S5  | 16.41408384079902 | 20.38570035418408 | 16.413172756532582 | 478.0390070489655 | 296.930752792446 | 295.204481832446 | 535.0402811142731 | 295.47280243634304 | 29.543193182073452 | 259.9306953789963 | 477.9714035932095 | 16.957302922645596 | 476.97921893358273 | 295.19027061286835 | 344.4023165803424 | 345.603099976026 | 311.0615468174578 | 16.31794206958455 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

(Continue for all suppliers S1–S18 and all customers C1–C18, using the exact coefficients as retrieved.)

---

**Complete Model:**

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\end{align*}
\]

Where all $d_j$, $s_i$, and $c_{ij}$ are as listed above.