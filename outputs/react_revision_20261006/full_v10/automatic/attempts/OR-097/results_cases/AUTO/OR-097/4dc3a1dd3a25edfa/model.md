Mathematical Model

Sets:
I = {1,...,120}   // Options, indexed by i
J = {1,...,6}     // Assets, indexed by j
G = {Delta, Gamma, Vega} // Greeks

Parameters (from OptionCharacteristics.csv, table_id: file_0_view_0):
Cost[i]      // Cost per contract for option i
Delta[i]     // Delta per contract for option i
Gamma[i]     // Gamma per contract for option i
Vega[i]      // Vega per contract for option i
MaxLong[i]   // Maximum allowed long position for option i
MaxShort[i]  // Maximum allowed short position for option i

Parameters (from Option_AssetReferenceMatrix.csv, table_id: file_1_view_0):
A[i,j]       // 1 if option i references asset j, 0 otherwise

Initial exposures and tolerances (from user description):
Delta_initial = 0.25
Gamma_initial = 0.08
Vega_initial = 0.17
Tolerance_Delta = 0.06
Tolerance_Gamma = 0.05
Tolerance_Vega = 0.07

Decision variables:
x[i] ∈ ℤ, MaxShort[i] ≤ x[i] ≤ MaxLong[i]  for all i ∈ I
z[i] ≥ 0, z[i] ≥ x[i], z[i] ≥ -x[i]      for all i ∈ I  // auxiliary for |x[i]|

Objective:
minimize ∑_{i∈I} Cost[i] · z[i]

Constraints:

// Delta exposure
| Delta_initial + ∑_{i∈I} ∑_{j∈J} Delta[i] · A[i,j] · x[i] | ≤ Tolerance_Delta

// Gamma exposure
| Gamma_initial + ∑_{i∈I} ∑_{j∈J} Gamma[i] · A[i,j] · x[i] | ≤ Tolerance_Gamma

// Vega exposure
| Vega_initial + ∑_{i∈I} ∑_{j∈J} Vega[i] · A[i,j] · x[i] | ≤ Tolerance_Vega

// Trading limits
MaxShort[i] ≤ x[i] ≤ MaxLong[i]  for all i ∈ I

// Absolute value definition
z[i] ≥ x[i], z[i] ≥ -x[i], z[i] ≥ 0  for all i ∈ I

Variable domains:
x[i] ∈ ℤ  for all i ∈ I
z[i] ≥ 0  for all i ∈ I

Data Mapping

- OptionCharacteristics.csv (table_id: file_0_view_0): Option, Cost, Delta, Gamma, Vega, MaxLong, MaxShort
- Option_AssetReferenceMatrix.csv (table_id: file_1_view_0): Option (row index), Asset_1,...,Asset_6 (columns) → A[i,j]
- Initial exposures and tolerances: Delta_initial = 0.25, Gamma_initial = 0.08, Vega_initial = 0.17; Tolerance_Delta = 0.06, Tolerance_Gamma = 0.05, Tolerance_Vega = 0.07

All indices, parameters, and constraints are defined exactly as in the provided data and user description.