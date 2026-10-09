Mathematical Model

Sets:
T = {0, 1, ..., 47}  // 48 half-hour intervals, each labeled by source_row in file_0_view_0
Let s_t = Requirement for interval t, from file_0_view_0 column "Requirement"
Let n = 48  // number of intervals
Let H = 16  // number of half-hour intervals in 8 hours

Decision variables:
x_i ∈ ℕ₀, for i ∈ T  // number of waitstaff starting work at interval i

Objective:
minimize  ∑_{i=0}^{n-1} x_i

Constraints:
For each t ∈ T:
  ∑_{i=0}^{n-1} a_{i,t} x_i ≥ s_t

where
a_{i,t} = 1 if interval t is covered by a shift starting at i, 0 otherwise.
A shift starting at i covers intervals {i, i+1, ..., i+H-1} modulo n (wrap-around 24 hours).

Formally,
a_{i,t} = 1 if (t - i) mod n ∈ {0, 1, ..., H-1}, else 0

Variable domains:
x_i ≥ 0 and integer, for all i ∈ T

Data Mapping:
Set T and parameter s_t are from file_0_view_0, columns "source_row" and "Requirement" in 44.csv.
Each interval t corresponds to source_row t, with time label from column "Time".
H = 16 (since 8 hours = 16 half-hour intervals).
All variables and constraints are indexed over T as defined above.