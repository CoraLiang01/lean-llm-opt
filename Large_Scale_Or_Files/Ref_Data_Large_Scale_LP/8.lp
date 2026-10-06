\ Model PortfolioOptimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xde15d504ebd1882d
Maximize
  73 investment[1] + 76 investment[2] + 65 investment[3] + 70 investment[4]
   + 68 investment[5] + 75 investment[6] + 67 investment[7]
   + 74 investment[8] + 69 investment[9] + 72 investment[10]
Subject To
 Total_Investment: investment[1] + investment[2] + investment[3]
   + investment[4] + investment[5] + investment[6] + investment[7]
   + investment[8] + investment[9] + investment[10] <= 35
 Risk_Market: 5.1 investment[1] + 6 investment[2] + 3.8 investment[3]
   + 5.3 investment[4] + 4.8 investment[5] + 6.7 investment[6]
   + 4 investment[7] + 5.8 investment[8] + 4.7 investment[9]
   + 5.6 investment[10] <= 42
 Risk_Credit: 6.2 investment[1] + 7.1 investment[2] + 5.7 investment[3]
   + 6.4 investment[4] + 6.1 investment[5] + 6.5 investment[6]
   + 5.8 investment[7] + 6.3 investment[8] + 6 investment[9]
   + 6.6 investment[10] <= 50
 Risk_Interest: 3.4 investment[1] + 3.8 investment[2] + 2.9 investment[3]
   + 3.2 investment[4] + 3.1 investment[5] + 4.1 investment[6]
   + 3 investment[7] + 3.7 investment[8] + 3.3 investment[9]
   + 3.5 investment[10] <= 29
 Risk_FX: 4.2 investment[1] + 4.6 investment[2] + 3.4 investment[3]
   + 4.8 investment[4] + 4 investment[5] + 5.1 investment[6]
   + 3.7 investment[7] + 4.9 investment[8] + 4.1 investment[9]
   + 4.5 investment[10] <= 39
 Risk_Liquidity: 2.8 investment[1] + 2.6 investment[2] + 2.4 investment[3]
   + 3.3 investment[4] + 2.9 investment[5] + 3.7 investment[6]
   + 2.5 investment[7] + 3 investment[8] + 2.8 investment[9]
   + 2.7 investment[10] <= 25
 Risk_IR_Risk_r: 5.9 investment[1] + 4.8 investment[2] + 6.2 investment[3]
   + 5.1 investment[4] + 4.9 investment[5] + 6.3 investment[6]
   + 5.6 investment[7] + 6.1 investment[8] + 5.2 investment[9]
   + 5.4 investment[10] <= 49
 Risk_Equity: 2.7 investment[1] + 2.5 investment[2] + 2.2 investment[3]
   + 2.6 investment[4] + 2.3 investment[5] + 3 investment[6]
   + 2.4 investment[7] + 2.9 investment[8] + 2.7 investment[9]
   + 2.8 investment[10] <= 25
 Risk_Volatility: 4.5 investment[1] + 5.1 investment[2] + 3.9 investment[3]
   + 4.8 investment[4] + 4.4 investment[5] + 5.4 investment[6]
   + 4 investment[7] + 5 investment[8] + 4.3 investment[9]
   + 4.7 investment[10] <= 36
 Risk_Regulatory: 2.2 investment[1] + 3 investment[2] + 2.7 investment[3]
   + 2.5 investment[4] + 2.4 investment[5] + 3.3 investment[6]
   + 2.6 investment[7] + 3.2 investment[8] + 2.3 investment[9]
   + 2.8 investment[10] <= 27
Bounds
 investment[1] <= 12
 investment[2] <= 8
 investment[3] <= 10
 investment[4] <= 9
 investment[5] <= 7
 investment[6] <= 11
 investment[7] <= 8
 investment[8] <= 10
 investment[9] <= 9
 investment[10] <= 12
End
