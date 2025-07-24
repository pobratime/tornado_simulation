import numpy as np

class BurgersVortex():
    def __init__(self):
      # ovo su velicine za veliku skalu
      self.alpha = 0.4 # Strain rate
      self.Gamma = 2000 # Circulation
      self.nu = 1.5e-6 # Kinematic viscosity
      
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
        return self.lambda_2_critetion_method(x, y, z)
      elif method == "liutex":
        return self.calculate_liutex_magnitude(x, y, z)
      elif method == "velocity_magnitude":
        vel = self.velocity2(x, y, z)
        return np.linalg.norm(vel)
      else:
        return 0.0
      
    def delta_method(self, x, y, z):
      try:
        grad = self.velocity_gradient(x, y, z)
        
        # Check for invalid gradients
        if not np.all(np.isfinite(grad)):
            return 0.0
            
        # Characteristic polynomial coefficients
        I1 = np.trace(grad)  # First invariant
        I2 = 0.5 * (I1**2 - np.trace(np.dot(grad, grad)))  # Second invariant (corrected formula)
        I3 = np.linalg.det(grad)  # Third invariant
        
        # Q and R for discriminant calculation
        Q = (3*I2 - I1**2) / 9
        R = (2*I1**3 - 9*I1*I2 + 27*I3) / 54
        
        # Discriminant delta
        delta = Q**3 + R**2
        
        # Return a more meaningful value for visualization
        # Use log scale for better distribution
        if delta > 0:
            return np.log10(1 + abs(delta))
        else:
            return -np.log10(1 + abs(delta))
            
      except Exception:
        return 0.0
      
      
    def Q_method(self, x, y, z):
      try:
        grad = self.velocity_gradient(x, y, z)
        
        # Check for invalid gradients
        if not np.all(np.isfinite(grad)):
            return 0.0
            
        # Strain rate tensor (symmetric part)
        S = 0.5 * (grad + grad.T) 
        # Vorticity tensor (antisymmetric part)
        Omega = 0.5 * (grad - grad.T)
        
        # Q-criterion: Q = 0.5 * (||Omega||^2 - ||S||^2)
        # Using Frobenius norm squared
        Omega_magnitude_sq = np.sum(Omega * Omega)
        S_magnitude_sq = np.sum(S * S)
        Q = 0.5 * (Omega_magnitude_sq - S_magnitude_sq)
        
        # Use logarithmic scaling for better visualization
        # Positive Q indicates vortex regions
        if Q > 0:
            return np.log10(1 + Q)
        else:
            return -np.log10(1 + abs(Q))
        
      except Exception:
        return 0.0
    
    def lambda_ci_criterion_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      eigvals = np.linalg.eigvals(grad)
      lambda_ci = np.max(np.abs(np.imag(eigvals)))
      
      return lambda_ci
      
    def lambda_2_critetion_method(self, x, y, z):
      grad = self.velocity_gradient(x, y, z)
      # Strain rate tensor (symmetric part)
      S = 0.5 * (grad + grad.T)
      # Vorticity tensor (antisymmetric part)  
      Omega = 0.5 * (grad - grad.T)
      
      # Lambda-2 criterion uses S^2 + Omega^2 tensor
      tensor = np.dot(S, S) + np.dot(Omega, Omega)
      eigvals = np.linalg.eigvals(tensor)
      eigvals_sorted = np.sort(eigvals)
      lambda_2 = eigvals_sorted[1]  # Second largest eigenvalue
      
      return lambda_2
    
    def calculate_alpha_beta(self, x, y, z):
      h = 1e-6
      du_dx = (self.velocity2(x + h, y, z)[0] - self.velocity2(x - h, y, z)[0]) / (2*h)
      du_dy = (self.velocity2(x, y + h, z)[0] - self.velocity2(x, y - h, z)[0]) / (2*h)
      dv_dx = (self.velocity2(x + h, y, z)[1] - self.velocity2(x - h, y, z)[1]) / (2*h)
      dv_dy = (self.velocity2(x, y + h, z)[1] - self.velocity2(x, y - h, z)[1]) / (2*h)

      # Corrected formulas for alpha and beta based on proper definitions
      alpha = 0.5 * np.sqrt((du_dx - dv_dy)**2 + (du_dy + dv_dx)**2)
      beta = 0.5 * (dv_dx - du_dy)

      return alpha, beta

    def calculate_liutex_magnitude(self, x, y, z):
      alpha, beta = self.calculate_alpha_beta(x, y, z)
      # Liutex magnitude: R = 2 * (|β| - α) where β > α
      # Only consider positive values (vortex regions)
      R = 2 * max(0, np.abs(beta) - alpha)
      
      return R
