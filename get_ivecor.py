def get_ivecor(N,F,S,T,tv_dim,ndim ,nmix):
  % N: 0th Baum-Welche statistics
  % F: 1st Bawm Welche statistics
  % S: Flattened diagonal covariance matrix of ubm
  % T: total variability space matrix (tvs)
  % tv_dim: dimension of tvs
  % ndim: feature dimention
  % nmix: number of gaussian components in ubm
  I = np.eye(tv_dim);
  T_invS =  T/ S.T
  idx_sv = np.reshape(np.matlib.repmat(range(0, nmix), ndim, 1), ndim * nmix, 1)
  L = I + ((T_invS* N[idx_sv].T).dot(T.T))
  B = np.dot(T_invS, F)
  ivector = np.dot(np.linalg.pinv(L) ,B)
  return ivector
