Mathematical Model

Sets:
Let 𝒞 be the set of Operations Research courses in courses_42.csv:
𝒞 = {C22, C23, C24, C25, C26, C27, C28}

Parameters (from table_id: file_0_view_0):
For each course c ∈ 𝒞:
- credits_c: number of credits for course c (column: credits)
- interest_c: interest points for course c (column: interest_points)

Decision Variables:
For each c ∈ 𝒞:
- x_c ∈ {0,1}, where x_c = 1 if course c is selected, 0 otherwise

Objective:
Maximize total interest points:
maximize ∑_{c∈𝒞} interest_c · x_c

Constraint:
Total credits of selected courses ≤ 20:
∑_{c∈𝒞} credits_c · x_c ≤ 20

Variable domains:
x_c ∈ {0,1} for all c ∈ 𝒞

Data Mapping:
- Set 𝒞: All rows in courses_42.csv with discipline = "Operations Research" (table_id: file_0_view_0, column: course_id)
- credits_c: table_id: file_0_view_0, column: credits, for each c
- interest_c: table_id: file_0_view_0, column: interest_points, for each c

Summary:
maximize ∑_{c∈𝒞} interest_c · x_c
subject to ∑_{c∈𝒞} credits_c · x_c ≤ 20
      x_c ∈ {0,1} ∀ c ∈ 𝒞
with all parameters and set 𝒞 defined by the filtered rows of courses_42.csv as above.