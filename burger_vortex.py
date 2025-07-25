import numpy as np

class BurgersVortex():
    def __init__(self, alpha=0.4, Gamma=2000, nu=1.5e-6):
      # ovo su velicine za veliku skalu
      self.alpha = alpha # Strain rate
      self.Gamma = Gamma # Circulation
      self.nu = nu # Kinematic viscosity
      
      # ovo su velicine za idealnu skalu na fludinom nivou
      self.alpha2 = 0.043
      self.Gamma2 = 1.45
      self.nu2 = 1.5e-6
      
    def velocity(self, x, y, z):
      r = np.sqrt(x**2 + y**2)
      r = np.where(r < 1e-10, 1e-10, r)
      
      v_r = - self.alpha * r
      v_z = 2 * self.alpha * z
      
      Re_local = self.alpha * r**2 / (2 * self.nu)
      v_theta = (self.Gamma / (2 * np.pi * r)) * (1 - np.exp(-Re_local))
      
      u_x = v_r * x/r - v_theta * y/r
      u_y = v_r * y/r + v_theta * x/r
      u_z = v_z
      
      return np.array([u_x, u_y, u_z])
    
    def velocity2(self, x, y, z):    
      r = np.sqrt(x**2 + y**2)
      r = np.where(r < 1e-10, 1e-10, r)
      
      v_r = - self.alpha2 * r
      v_z = 2 * self.alpha2 * z
      
      Re_local = self.alpha2 * r**2 / (2 * self.nu2)
      v_theta = (self.Gamma2 / (2 * np.pi * r)) * (1 - np.exp(-Re_local))
      
      u_x = v_r * x/r - v_theta * y/r
      u_y = v_r * y/r + v_theta * x/r
      u_z = v_z
      
      return np.array([u_x, u_y, u_z]) 
     
    def velocity_gradient(self, x, y, z):
      h = 1e-6
      grad = np.zeros((3,3))
      
      grad[0,0] = (self.velocity2(x + h, y, z)[0] - self.velocity2(x - h, y, z)[0]) / (2*h)
      grad[0,1] = (self.velocity2(x, y + h, z)[0] - self.velocity2(x, y - h, z)[0]) / (2*h)
      grad[0,2] = (self.velocity2(x, y, z + h)[0] - self.velocity2(x, y, z - h)[0]) / (2*h)
      
      grad[1,0] = (self.velocity2(x + h, y, z)[1] - self.velocity2(x - h, y, z)[1]) / (2*h)
      grad[1,1] = (self.velocity2(x, y + h, z)[1] - self.velocity2(x, y - h, z)[1]) / (2*h)
      grad[1,2] = (self.velocity2(x, y, z + h)[1] - self.velocity2(x, y, z - h)[1]) / (2*h)
      
      grad[2,0] = (self.velocity2(x + h, y, z)[2] - self.velocity2(x - h, y, z)[2]) / (2*h)
      grad[2,1] = (self.velocity2(x, y + h, z)[2] - self.velocity2(x, y - h, z)[2]) / (2*h)
      grad[2,2] = (self.velocity2(x, y, z + h)[2] - self.velocity2(x, y, z - h)[2]) / (2*h)
        
      return grad
      
      
    def return_coloring(self, x, y, z, method):
      if method == "delta":
        return self.delta_method(x, y, z)
      elif method == "Q":
        return self.Q_method(x, y, z)
      elif method == "lambda_ci":
        return self.lambda_ci_criterion_method(x, y, z)
      elif method == "lambda_2":
        return self.lambda_2_criterion_method(x, y, z)
      elif method == "liutex":
        return self.calculate_liutex_magnitude(x, y, z)
      elif method == "velocity_magnitude":
        vel = self.velocity2(x, y, z)
        return np.linalg.norm(vel)
      else:
        return 0.0
      
    def lambda_2_criterion_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      
      if not np.all(np.isfinite(grad)):
          return 0.0
      
      S = 0.5 * (grad + grad.T)
      Omega = 0.5 * (grad - grad.T)
      
      tensor = np.dot(S, S) + np.dot(Omega, Omega)
      eigvals = np.sort(np.linalg.eigvals(tensor))
      
      return eigvals[1] 
      
    def delta_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      
      if not np.all(np.isfinite(grad)):
          return 0.0
          
      I1 = np.trace(grad)
      I2 = 0.5 * (I1**2 - np.trace(np.dot(grad, grad)))
      I3 = np.linalg.det(grad)
      
      Q = (3*I2 - I1**2) / 9
      R = (2*I1**3 - 9*I1*I2 + 27*I3) / 54
      
      delta = Q**3 + R**2
      
      return np.log10(1 + abs(delta)) * np.sign(delta)
      
    def Q_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      
      if not np.all(np.isfinite(grad)):
          return 0.0
          
      S = 0.5 * (grad + grad.T) 
      Omega = 0.5 * (grad - grad.T)
      
      Q = 0.5 * (np.sum(Omega * Omega) - np.sum(S * S))
      
      return np.log10(1 + abs(Q)) * np.sign(Q)

    def lambda_ci_criterion_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      
      if not np.all(np.isfinite(grad)):
          return 0.0
      
      eigvals = np.linalg.eigvals(grad)
      return np.max(np.abs(np.imag(eigvals)))
    
    def calculate_alpha_beta(self, x, y, z):
      h = 1e-6
      du_dx = (self.velocity2(x + h, y, z)[0] - self.velocity2(x - h, y, z)[0]) / (2*h)
      du_dy = (self.velocity2(x, y + h, z)[0] - self.velocity2(x, y - h, z)[0]) / (2*h)
      dv_dx = (self.velocity2(x + h, y, z)[1] - self.velocity2(x - h, y, z)[1]) / (2*h)
      dv_dy = (self.velocity2(x, y + h, z)[1] - self.velocity2(x, y - h, z)[1]) / (2*h)

      alpha = 0.5 * np.sqrt((du_dx - dv_dy)**2 + (du_dy + dv_dx)**2)
      beta = 0.5 * (dv_dx - du_dy)

      return alpha, beta

    def calculate_liutex_magnitude(self, x, y, z):
      alpha, beta = self.calculate_alpha_beta(x, y, z)
      R = 2 * max(0, np.abs(beta) - alpha)
      return R
