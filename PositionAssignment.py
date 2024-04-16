
def position_assign(P,Q,M,N,R,C,i,j):
  """
  Simple position assign. Mapping strategy: Matrix (RxC) -> (MP)x(NQ)
  Assumption: 
    1. A memristor crossbar array device contains an (MxN) number of chiplets, 
       each has an(PxQ) number of cells.
    2. For simplicity, R = C, while R = M*P and C = N * Q.   

  (P,Q): Dimension of memristor crossbar array unit.
  (M,N): Dimension of memristor crossbar array chiplet.
  (R,C): Dimension of the matrix.
  """
  assert(R == M*P), "We cannot map this matrix to the device"
  assert(C == N*Q), "We cannot map this matrix to the device"

  # Assign the position based on the chiplet's index 
  # For example, (i=3,j=3) means the chiplet located at Row 2 and Column 3
  true_row = i - 1; true_column = j
  if (true_row == 0):
    s_r = 0; e_r = P
  else:
    s_r = true_row * P; e_r = true_row * P + P
  if (true_column == 0):
    s_c = 0; e_c = Q
  else:
    s_c = true_column * Q, true_column * Q + Q

  return s_c, e_c, s_r, e_r