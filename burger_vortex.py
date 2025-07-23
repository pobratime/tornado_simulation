import numpy as np

class BurgersVortex():
    def __init__(self):
      self.alpha = 0.042 # Strain rate
      self.Gamma = 1.45 # Circulation
      self.nu = 0.01 # Kinematic viscosity
      
    def velocity(self, x, y, z):
      """
      Racunamo komponente brzine Burgerovog vihora u cilindricnim koordinatama
      Nakon toga ih prebacujemo u Dekartov pravougli koordinatni sistem
      """
      # distanca od z ose
      r = np.sqrt(x**2 + y**2)
      r = np.where(r < 1e-10, 1e-10, r)
      
      v_r = - self.alpha * r
      v_z = 2 * self.alpha * z
      # rejndsolov broj
      Re_local = self.alpha * r**2 / (2 * self.nu)
      v_theta = (self.Gamma / (2 * np.pi * r)) * (1 - np.exp(-Re_local))
      
      u_x = v_r * x/r - v_theta * y/r
      u_y = v_r * y/r + v_theta * x/r
      u_z = v_z
      
      return np.array([u_x, u_y, u_z])
    
    def velocity_gradient(self, x, y, z):
      h = 1e-6
      grad = np.zeros((3,3))
      
      grad[0,0] = (self.velocity(x + h, y, z)[0] - self.velocity(x - h, y, z)[0]) / (2*h)
      grad[0,1] = (self.velocity(x, y + h, z)[0] - self.velocity(x, y - h, z)[0]) / (2*h)
      grad[0,2] = (self.velocity(x, y, z + h)[0] - self.velocity(x, y, z - h)[0]) / (2*h)
      
      grad[1,0] = (self.velocity(x + h, y, z)[1] - self.velocity(x - h, y, z)[1]) / (2*h)
      grad[1,1] = (self.velocity(x, y + h, z)[1] - self.velocity(x, y - h, z)[1]) / (2*h)
      grad[1,2] = (self.velocity(x, y, z + h)[1] - self.velocity(x, y, z - h)[1]) / (2*h)
      
      grad[2,0] = (self.velocity(x + h, y, z)[2] - self.velocity(x - h, y, z)[2]) / (2*h)
      grad[2,1] = (self.velocity(x, y + h, z)[2] - self.velocity(x, y - h, z)[2]) / (2*h)
      grad[2,2] = (self.velocity(x, y, z + h)[2] - self.velocity(x, y, z - h)[2]) / (2*h)
        
      return grad
      
    def delta_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      I1 = np.trace(grad)
      I2 = -1/2 * (np.trace(grad)**2 - np.trace(np.dot(grad, grad)))
      I3 = np.linalg.det(grad)
      
      Q = I2 - (1/3) * I1**2
      R = -I3 - (2/27) * I1**3 + (1/3) * I1 * I2
      
      delta = (Q/3)**3 + (R/2)**2
      
      return delta
      
      
    def Q_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      S = 0.5 * (grad + grad.T) 
      Omega = 0.5 * (grad - grad.T)
      Q = 0.5 * (np.linalg.norm(Omega, 'fro')**2 - np.linalg.norm(S, 'fro')**2)
      
      return Q
    
    def lambda_ci_criterion_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      eigvals = np.linalg.eigvals(grad)
      lambda_ci = np.max(np.abs(np.imag(eigvals)))
      
      return lambda_ci
      
    def lambda_2_critetion_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      S = 0.5 * (grad + grad.T)
      eigvals = np.linalg.eigvals(S)
      eigvals_sorted = np.sort(eigvals)
      lambda_2 = eigvals_sorted[1]
      
      return lambda_2
    
    def calculate_alpha_beta(self, x, y, z):
      h = 1e-6
      du_dx = (self.velocity(x + h, y, z)[0] - self.velocity(x - h, y, z)[0]) / (2 * h)
      du_dy = (self.velocity(x, y + h, z)[0] - self.velocity(x, y - h, z)[0]) / (2 * h)
      dv_dx = (self.velocity(x + h, y, z)[1] - self.velocity(x - h, y, z)[1]) / (2 * h)
      dv_dy = (self.velocity(x, y + h, z)[1] - self.velocity(x, y - h, z)[1]) / (2 * h)

      alpha = 0.5 * np.sqrt((dv_dy - du_dx)**2 + (dv_dx + du_dy)**2)
      beta = 0.5 * (dv_dx - du_dy)

      return alpha, beta

    def calculate_liutex_magnitude(self, x, y, z):
      alpha, beta = self.calculate_alpha_beta(x, y, z)
      if(beta**2 > alpha**2):
        R = 2 * (np.abs(beta) - alpha)
      else:
        R = 0
        
      return R
    
    def calculate_liutex_magnitude2(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)

      omega = np.array([
          grad[2,1] - grad[1,2],
          grad[0,2] - grad[2,0],
          grad[1,0] - grad[0,1]
      ])

      return abs(omega[2])