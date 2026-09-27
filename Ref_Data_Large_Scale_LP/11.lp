\ Model MaximizeProducts
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xaebb8f5525e3d921
Maximize
  N
Subject To
 Hours_A: x[A,1] + x[A,2] + x[A,3] <= 100
 Hours_B: x[B,1] + x[B,2] + x[B,3] <= 150
 Hours_C: x[C,1] + x[C,2] + x[C,3] <= 80
 Hours_D: x[D,1] + x[D,2] + x[D,3] <= 200
 Requirement_1: - N + 10 x[A,1] + 15 x[B,1] + 20 x[C,1] + 10 x[D,1] >= 0
 Requirement_2: - N + 15 x[A,2] + 10 x[B,2] + 5 x[C,2] + 15 x[D,2] >= 0
 Requirement_3: - N + 5 x[A,3] + 5 x[B,3] + 10 x[C,3] + 20 x[D,3] >= 0
Bounds
End
