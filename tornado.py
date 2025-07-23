import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGridLayout, QPushButton, QHBoxLayout, QLabel, QLineEdit
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl
import pyqtgraph as pq

import burger_vortex as bv

# TODO TODO TODO TODO TODO
# magnus effect -> for projectile rotation
# swirl ratio
# add mulitple projectiles maybe?
# add user input tab
# improve/fix air resistance?
# check acceleration for bugs
# check for bugs in general
# Burgers vertex DONE
# USER INPUT FIX
# ADD GENERAL PLOTTING TAB
# TODO TODO TODO TODO TODO

class Sphere():
    def __init__(self):
        """
        OSNOVNI PODACI O PROJEKTILU
        
        Kao projektil u ovom slucaju koristmo sferu jer je najalaksa za sada.
        
        Parametri:
        position -> trenutna pozicija projektila prostoru kao :d vektor
        velocity -> trenutna brzina projektila (jer brzina u sva tri pravca) kao 3d vektor
        acceleration -> trenutno ubrzanje projektila kao 3d vektor
        mass -> masa projektila
        radius -> poluprecnik objekta
        diameter -> precnik objekta
        """
        
        self.position = np.array([0, 3, 30.0], dtype=float)
        self.velocity = np.array([0, 0, 0], dtype=float)
        self.acceleration = np.array([0, 0, 0], dtype=float)
        self.mass = 0.2
        self.radius = 0.5
        self.diameter = 2 * self.radius
        
class Tornado():
    def __init__(self):
        """
        OSNOVNI PODACI O TORNADU
        
        Kao tornado u ovom projektu koristimo tornado cilindricnog oblika, tj. tornado koji je iste sirine pri dnu i vrhu, ne klasicni tornado V oblika.
        
        Parametri:
        projectile -> projektil koji koristimo
        radius -> poluprecnik tornada
        diameter -> precnik tornada
        gravity -> gravitaciona konstana 9.81 kao 3d vektor zarad racunanja
        max_speed_horizontal -> maksimalna brzina vetra po horizontali
        max_speed_vertical -> maksimalna brzina vetra po vertikali
        inflow_angle ->
        K ->
        """
        
        self.projectile = Sphere()
        self.radius = 10
        self.diameter = 2 * self.radius
        self.rho = 1.293
        self.gravity = np.array([0, 0, -9.81], dtype=float) # -9.81 m/s !
        self.max_speed_horizontal = 150 
        self.max_speed_vertical = 150
        
        self.bv = bv.BurgersVortex()
        
        self.inflow_angle = np.deg2rad(0)
        self.K = 0.5

    def is_inside_tornado(self):
        x, y, _ = self.projectile.position
        r = np.sqrt(x**2 + y**2)
        return r <= self.radius

    def calculate_magnus_effect(self):
        """
        MAGNUS EFFECT
        
        Parametri:
        
        Racunanje:
        
        """
        # TOOD
        return np.array([0, 0, 0])

    def calculate_air_resistance(self, wind_velocity):
        """
        RACUNANJE OTPORA VAZDUHA
        
        Parametri:
        v -> trenutna brzina projektila
        d -> precnik projektila
        rho -> gustina vazduha
        Cd -> otpor (0.47 ZA SFERU!)
        D 
        Racunanje:
        
        """
        
        v = self.projectile.velocity
        d = self.projectile.diameter
        rho = self.rho
        
        Cd = 0.47
        D = 0.5 * Cd * rho * np.pi * (d/2)**2 
        # D = 3.0 * rho * d**2
        air_resistance = -D * (v - wind_velocity) * np.linalg.norm(v - wind_velocity)
        
        return air_resistance

    def calculate_tornado_wind_velocity(self):
        """
        BURGER-ROTT VERTEX MODEL
        TODO
        PROVERITI OVO I NAMESTITI OSTATAK
        """
        x, y, z = self.projectile.position
        velocity = self.bv.velocity(x, y, z)
        
        return velocity
         
    def calulate_acceleration(self):
        """
        RACUNANJE UBZANJA SFERE
        
        Parametri:
        
        Racuanje:
        
        """
        
        wind_velocity = self.calculate_tornado_wind_velocity()
        D = self.calculate_air_resistance(wind_velocity)
        m = self.projectile.mass
        g = self.gravity

        a = D / m + g
        
        return a
    
    def calculate_core_radius(self):
        """
        RACUNANJE CENTRA TORNADA
        
        Parametri:
        
        Racuanje:
        
        """
        sin2_theta = np.sin(self.inflow_angle)**2
        r_outer = self.radius
        numerator = r_outer * sin2_theta
        denominator = 1 - self.K * sin2_theta
        r_inner = numerator / denominator
        
        return r_inner
    
    def update_projectile(self, delta_time):
        """
        AZURIRANJE PODATAKA KOJI SE MENJAJU U REALNOM VREMENU
        """
        self.projectile.acceleration = self.calulate_acceleration()
        self.projectile.velocity += self.projectile.acceleration * delta_time
        self.projectile.position += self.projectile.velocity * delta_time


"""
==============================================================================================================================================================================
SVE STO SE NALAZI ISPOD OVE LINIJE JE KOD TEHNICKE PRIRODE KOJI SE KORISTI ZA PRIKAZIVANJE I RENDEROVANJE I NIJE RELEVANTAN ZA RACUNANJE I OSTALE STVARI TE NIJE DOKUMENTIRAN
==============================================================================================================================================================================
"""

class KinematicSimulationTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 360
        self.number_of_particles = 500
        self.positions = np.random.uniform(-1, 1, (self.number_of_particles, 3))
        self.vortex = bv.BurgersVortex()
        self.scatter = gl.GLScatterPlotItem(pos=self.positions, size=5, color=(0.2, 0.8, 1.0, 1.0), pxMode=True)
        self.view.addItem(self.scatter)
        layout.addWidget(self.view)  # Add this line, missing in your code

    def update_kinematic(self, delta_time):
        for i in range(self.number_of_particles):
            vel = self.vortex.velocity(*self.positions[i])
            self.positions[i] += vel * delta_time
            if np.linalg.norm(self.positions[i]) > 2.5 or abs(self.positions[i][2]) > 2.5:
                self.positions[i] = np.random.uniform(-1, 1, 3)
        self.scatter.setData(pos=self.positions)
        

class TornadoSimulationTab(QWidget):
    def __init__(self, tornado):
        super().__init__()
        layout = QVBoxLayout(self)
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 360

        grid = gl.GLGridItem(antialias=True)
        grid.setSize(200, 200)
        grid.setSpacing(10, 10, 10)
        grid.translate(0, 0, -0.1)
        self.view.addItem(grid)

        axis = gl.GLAxisItem(antialias=True)
        axis.setSize(100, 100, 100)
        self.view.addItem(axis)

        axisneg = gl.GLAxisItem(antialias=True)
        axisneg.setSize(-100, -100, -100)
        self.view.addItem(axisneg)

        self.line = gl.GLLinePlotItem(color=(1, 1, 0, 1), width=2, antialias=True)
        self.view.addItem(self.line)

        self.point = gl.GLScatterPlotItem(size=10, color=(1, 0, 0, 1))
        self.view.addItem(self.point)

        layout.addWidget(self.view)

        self.tornado = tornado
        self.positions = []

    # TODO
    def update_simulation(self, delta_time):
        if self.tornado.projectile.position[2] > 0.0:
            self.tornado.update_projectile(delta_time)
            pos = self.tornado.projectile.position.copy()
            self.positions.append(pos)
            pts = np.array(self.positions)
            self.line.setData(pos=pts)
            self.point.setData(pos=np.array([pos]))
        else:
            pass

class PlottingTab(QWidget):
    def __init__(self, tornado, titles, data_getter):
        super().__init__()
        layout = QGridLayout(self)
        self.setLayout(layout)

        self.plots = []
        self.curves = []
        self.times = []
        self.data_series = [[] for _ in range(3)]
        self.data_series_magnitude = []

        for i, title in enumerate(titles):
            plot = pq.PlotWidget(title=title)
            curve = plot.plot(pen=['r', 'g', 'b'][i])
            layout.addWidget(plot, 0, i)
            self.plots.append(plot)
            self.curves.append(curve)

        plot_mag = pq.PlotWidget(title="Magnitude")
        curve_mag = plot_mag.plot(pen='y')
        layout.addWidget(plot_mag, 1, 0, 1, 3)
        self.plots.append(plot_mag)
        self.curves.append(curve_mag)

        self.tornado = tornado
        self.data_getter = data_getter

    def update_plots(self, time):
        data = self.data_getter()
        self.times.append(time)
        for i in range(3):
            self.data_series[i].append(data[i])
        mag = np.linalg.norm(data)
        self.data_series_magnitude.append(mag)

        for i in range(3):
            self.curves[i].setData(self.times, self.data_series[i])
        self.curves[3].setData(self.times, self.data_series_magnitude)

class ControlPanel(QWidget):
    def __init__(self, tornado):
        super().__init__()
        layout = QHBoxLayout(self)
        self.tornado = tornado
        
        projectile_layout = QVBoxLayout()
        projectile_layout.addWidget(QLabel("Projectile details"))
        
        pos_input = QHBoxLayout()
        pos_input.addWidget(QLabel("Projectile starting postition (x, y, z)"))
        for i in range(3):
            le = QLineEdit("0")
            pos_input.addWidget(le)
            
        vel_input = QHBoxLayout()
        vel_input.addWidget(QLabel("Projectile starting velocity (x, y, z)"))
        for i in range(3):
            le = QLineEdit("0")
            vel_input.addWidget(le)
            
        projectile_layout.addLayout(pos_input)
        projectile_layout.addLayout(vel_input)
        
        mass_input_layout = QHBoxLayout()
        mass_input_layout.addWidget(QLabel("Projectile mass in kg"))
        mass_input = QLineEdit("0")
        mass_input_layout.addWidget(mass_input)
       
        radius_input_layout = QHBoxLayout()
        radius_input_layout.addWidget(QLabel("Projectile radius in m"))
        radius_input = QLineEdit("0")
        radius_input_layout.addWidget(radius_input)
        
        projectile_layout.addLayout(mass_input_layout)
        projectile_layout.addLayout(radius_input_layout)
        
        tornado_layout = QVBoxLayout()
        tornado_layout.addWidget(QLabel("Tornado details"))
        
        alpha_input_layout = QHBoxLayout()
        alpha_input_layout.addWidget(QLabel("Alpha"))
        alpha_input = QLineEdit("0")
        alpha_input_layout.addWidget(alpha_input) 
        tornado_layout.addLayout(alpha_input_layout)
        
        nu_input_layout = QHBoxLayout()
        nu_input_layout.addWidget(QLabel("Gamma"))
        nu_input = QLineEdit("0")
        nu_input_layout.addWidget(nu_input)
        tornado_layout.addLayout(nu_input_layout)
        
        gamma_input_layout = QHBoxLayout()
        gamma_input_layout.addWidget(QLabel("Gamma"))
        gamma_input = QLineEdit("0")
        gamma_input_layout.addWidget(gamma_input)
        tornado_layout.addLayout(gamma_input_layout)
        
        inflow_angle_input_layout = QHBoxLayout()
        inflow_angle_input_layout.addWidget(QLabel("Inflow angle"))
        inflow_angle_input = QLineEdit("BROKEN")
        inflow_angle_input_layout.addWidget(inflow_angle_input)
        tornado_layout.addLayout(inflow_angle_input_layout)
       
        
        self.launch_button = QPushButton("Launch Projectile")
        self.launch_button.clicked.connect(self.launch_projectile)

        layout.addLayout(projectile_layout)
        layout.addLayout(tornado_layout)
        layout.addWidget(self.launch_button)

    def launch_projectile(self):
        mainWindow = self.parent()
        while mainWindow and not isinstance(mainWindow, QMainWindow):
            mainWindow = mainWindow.parent()

        mainWindow.stop_update()

        if not hasattr(mainWindow, 'timer'):
            mainWindow.timer = QTimer()
            mainWindow.timer.timeout.connect(mainWindow.update_all)
            mainWindow.timer.start(16)
            mainWindow.time = 0.0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Tornado with 3D simulation and plotting')
        self.resize(640, 480)

        self.tornado = Tornado()

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.kinematic_tab = KinematicSimulationTab()
        self.sim_tab = TornadoSimulationTab(self.tornado)
        self.plot_tab = PlottingTab(self.tornado, ["vx", "vy", "vz"], lambda: self.tornado.projectile.velocity)
        self.plot_tab_acc = PlottingTab(self.tornado, ["ax", "ay", "az"], lambda: self.tornado.projectile.acceleration)

        self.tabs.addTab(self.sim_tab, "Tornado Simulation")
        self.tabs.addTab(self.kinematic_tab, "KinematicTab")
        self.tabs.addTab(self.plot_tab, "Tornado Plotting Vel")
        self.tabs.addTab(self.plot_tab_acc, "Tornado Plotting Acc")

        self.control_panel = ControlPanel(self.tornado)
        main_layout.addWidget(self.control_panel)
        
        self.setCentralWidget(main_widget)
    
    def update_tornado(self):
        self.tornado = Tornado()
        self.sim_tab.tornado = self.tornado
        self.plot_tab.tornado = self.tornado
        self.plot_tab_acc.tornado = self.tornado

    def update_all(self):
        if self.tornado.projectile.position[2] > 0:
            delta_time = 0.01
            self.sim_tab.update_simulation(delta_time)
            self.time += delta_time
            self.plot_tab.update_plots(self.time)
            self.plot_tab_acc.update_plots(self.time)
            self.kinematic_tab.update_kinematic(self.time)
            
        else:
            self.timer.stop()
    
    def stop_update(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer

        self.time = 0.0
        self.sim_tab.positions = []
        self.sim_tab.line.setData(pos=np.empty((0, 3)))
        self.sim_tab.point.setData(pos=np.empty((0, 3)))
        self.update_tornado()

        self.plot_tab.times = []
        self.plot_tab.data_series = [[] for _ in range(3)]
        self.plot_tab.data_series_magnitude = []
        for curve in self.plot_tab.curves:
            curve.setData([], [])

        self.plot_tab_acc.times = []
        self.plot_tab_acc.data_series = [[] for _ in range(3)]
        self.plot_tab_acc.data_series_magnitude = []
        for curve in self.plot_tab_acc.curves:
            curve.setData([], [])

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()