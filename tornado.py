import sys
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl

class Sphere():
    def __init__(self):
        """OSNOVNI PODACI O SFERI"""
        # TODO DA KORISNIK MOZE ODABRATI OVE STVARI
        self.position = np.array([0, 0, 1.0], dtype=float) # pozicija sfere u trenutku
        self.velocity = np.array([0, 0, 100 ], dtype=float) # brzina sfere
        self.acceleration = np.array([0, 0, 0], dtype=float) # ubrzanje sfere
        self.mass = 5 # masa sfere u kg
        self.radius = 2 # poluprecnik sfere u m
        self.diameter = 2 * self.radius # precnik sfere
        
class Tornado():
    def __init__(self):
        """OSNOVNI PODACI O TORNADU"""
        # TODO DA KORISNIK MOZE ODABRATI OVE STVARI
        self.projectile = Sphere()
        self.radius = 5
        self.diameter = 2 * self.radius
        self.rho = 1.293
        self.gravity = np.array([0, 0, -9.81], dtype=float)
        self.max_speed = 150

    def calculate_air_resistance(self, wind_velocity):
        """RACUANJE OTPORA VAZDUHA"""
        
        v = self.projectile.velocity
        d = self.projectile.diameter
        rho = self.rho
        
        Cd = 0.47
        D = 0.5 * Cd * rho * np.pi * (d/2)**2 
        air_resistance = -D * (v - wind_velocity) * np.abs(v - wind_velocity)
        
        return air_resistance

    def calculate_tornado_wind_velocity(self):
        """RACUNANJE BRZINE VETRRA UNUTAR TORNADA"""
        
        x, y, _ = self.projectile.position
        r = np.sqrt(x**2 + y**2)
        r_s = self.radius
        u0 = self.max_speed
        
        if r == 0:
            return np.zeros(3)
        if r > self.radius:
            speed = u0 * (r_s / r)
        else:
            speed = u0 * (r / r_s)
            
        u_theta = np.array([-y/r, x/r, 0.0])
        u = speed * u_theta
        
        return u
      
    def calulate_acceleration(self):
        """RACUNANJE UBZANJA SFERE"""
        wind_velocity = self.calculate_tornado_wind_velocity()
        D = self.calculate_air_resistance(wind_velocity)
        m = self.projectile.mass
        g = self.gravity

        a = D / m + g
        
        return a
    
    def update_projectile(self, delta_time):
        """AZURIRANJE PODATAKA KOJI SE MENJAJU U REALNOM VREMENU"""
        """U OVO SPADAJU BRZINA SFERE, UBRZANJE SFERE I POZICIJA SFERE"""
        a = self.calulate_acceleration()
        # print("acc : ", a, " velocity: ", self.projectile.velocity, " pos: ", self.projectile.position)
        self.projectile.velocity += a * delta_time
        self.projectile.position += self.projectile.velocity * delta_time
        
        
class App():
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Tornado')
        self.window.resize(800, 600)
        self.window.opts['distance'] = 40

        grid = gl.GLGridItem()
        grid.setSize(100, 100, 20)
        grid.setSpacing(1, 1, 1)
        grid.translate(0, 0, -0.01) 
        self.window.addItem(grid)

        axis = gl.GLAxisItem()
        axis.setSize(20, 20, 20)
        self.window.addItem(axis)
        
        axisneg = gl.GLAxisItem()
        axisneg.setSize(-20, -20, -20)
        self.window.addItem(axisneg)
        
        self.tornado = Tornado()
        self.positions = []

        self.line = gl.GLLinePlotItem(color=(1, 1, 0, 1), width=2, antialias=True)
        self.window.addItem(self.line)

        self.point = gl.GLScatterPlotItem(size=10, color=(1, 0, 0, 1))
        self.window.addItem(self.point)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    def update(self):
        delta_time = 0.01
        if(self.tornado.projectile.position[2] > 0.0):
          self.tornado.update_projectile(delta_time)
          pos = self.tornado.projectile.position.copy()

          self.positions.append(pos)

          pts = np.array(self.positions)
          self.line.setData(pos=pts)

          self.point.setData(pos=np.array([pos]))
        else:
          self.timer.stop()
          
    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())
    

def main():
    app = App()
    app.run()

if __name__ == "__main__":
    main()