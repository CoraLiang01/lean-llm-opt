Mathematical Model

Sets:
  Let 𝒞 be the set of Operations Research courses in courses_42.csv:
    𝒞 = {C22, C23, C24, C25, C26, C27, C28}

Parameters (from file_0_view_0):
  For each course c ∈ 𝒞:
    credits_c = [credits column, integer]
    interest_c = [interest_points column, integer]

Decision Variables:
  For each c ∈ 𝒞:
    x_c ∈ {0,1}   (1 if course c is selected, 0 otherwise)

Objective:
  Maximize total interest points:
    maximize   ∑_{c ∈ 𝒞} interest_c * x_c

Constraint:
  Total credits of selected courses ≤ 20:
    ∑_{c ∈ 𝒞} credits_c * x_c ≤ 20

Variable domains:
  x_c ∈ {0,1}   for all c ∈ 𝒞

Data Mapping:
  - Set 𝒞 and parameters credits_c, interest_c are taken from file_0_view_0 (courses_42.csv), filtered where discipline = "Operations Research", using columns:
      course_id → c
      credits → credits_c
      interest_points → interest_c

  - The model uses all 7 Operations Research courses in the file.

Summary:
  maximize   ∑_{c ∈ 𝒞} interest_c * x_c
  subject to ∑_{c ∈ 𝒞} credits_c * x_c ≤ 20
             x_c ∈ {0,1}  ∀ c ∈ 𝒞

  with all parameters and set members mapped directly from file_0_view_0 in courses_42.csv.