\ Model MaterialBlending
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x74b58b452f0412a5
Maximize
  - 5 x[A,1] + 35 x[A,2] + 25 x[A,3] + 5 x[B,1] + 45 x[B,2] + 35 x[B,3]
   - 5 x[C,1] + 35 x[C,2] + 25 x[C,3]
Subject To
 Material_Availability[1]: x[A,1] + x[B,1] + x[C,1] <= 100
 Material_Availability[2]: x[A,2] + x[B,2] + x[C,2] <= 100
 Material_Availability[3]: x[A,3] + x[B,3] + x[C,3] <= 60
 ProdA_Spec1: 0.5 x[A,1] - 0.5 x[A,2] - 0.5 x[A,3] >= 0
 ProdA_Spec2: - 0.25 x[A,1] + 0.75 x[A,2] - 0.25 x[A,3] <= 0
 ProdB_Spec1: 0.75 x[B,1] - 0.25 x[B,2] - 0.25 x[B,3] >= 0
 ProdB_Spec2: - 0.5 x[B,1] + 0.5 x[B,2] - 0.5 x[B,3] <= 0
Bounds
End
