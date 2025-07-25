import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGridLayout, QPushButton, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QListWidget, QComboBox
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QDoubleValidator
from pyvistaqt import QtInteractor
import pyqtgraph.opengl as gl
import pyqtgraph as pq
import pyvista as pv
from tqdm import tqdm
import burger_vortex as bv
import math

show_liutex_tab = False
liutex_settings_selected = None
liutex_settings = {
    "Low": (4, 25),
    "Medium": (4, 100),
    "High": (4, 300),
    "Very High": (4, 1000)
}

user_data_settings = {
    "position": [0.0, 3.0, 30.0],
    "velocity": [0.0, 0.0, 0.0],
    "mass": 20.0,
    "radius": 3.0,
    "alpha": 0.1,
    "nu": 1.5e-5,
    "gamma": 2000
}

class Sphere():
    
    def __init__(self, position=[0, 3, 30.0], velocity=[0, 0, 0], mass=20, radius=3):
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
        
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.array([0, 0, 0], dtype=float)
        self.mass = mass
        self.radius = radius
        self.diameter = 2 * self.radius
        
class Tornado():
    def __init__(self, position=[0, 3, 30.0], velocity=[0, 0, 0], mass=20, radius=3, alpha = 0.1, nu = 1.5e-5, gamma = 2000):
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
        
        self.projectile = Sphere(position, velocity, mass, radius)
        self.radius = 10
        self.diameter = 2 * self.radius
        self.rho = 1.293
        self.gravity = np.array([0, 0, -9.81], dtype=float) # -9.81 m/s !
        self.max_speed_horizontal = 150 
        self.max_speed_vertical = 150
        
        self.alpha = alpha
        self.nu = nu
        self.gamma = gamma
        
        self.bvortex = bv.BurgersVortex(self.alpha, self.gamma, self.nu)
        
        self.inflow_angle = np.deg2rad(0)
        self.K = 0.5

    def calculate_magnus_effect(self):
        """
        MAGNUS EFFECT
        
        Returns the Magnus force as a vector.
        
        Uses:
        - Cross product of angular velocity and relative velocity
        - Sphere's radius and air density
        - Magnus coefficient (empirical)
        """
        v_rel = self.projectile.velocity - self.calculate_tornado_wind_velocity()

        omega = np.array([0.0, 0.0, 50.0]) 

        C_m = 0.2

        rho = self.rho
        r = self.projectile.radius

        magnus_force = C_m * rho * r**3 * np.cross(omega, v_rel)
        
        return magnus_force

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
        return self.bvortex.velocity(x, y, z)
         
    def calulate_acceleration(self):
        """
        RACUNANJE UBZANJA SFERE
        
        Parametri:
        
        Racuanje:
        
        """
        
        wind_velocity = self.calculate_tornado_wind_velocity()
        D = self.calculate_air_resistance(wind_velocity)
        M = self.calculate_magnus_effect()
        m = self.projectile.mass
        g = self.gravity

        a = (D+M) / m + g
        
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

class particleSimulationTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        
        controls_layout = QHBoxLayout()
        
        self.add_particles_btn = QPushButton("Add Particles")
        self.add_particles_btn.clicked.connect(self.add_random_particles)
        controls_layout.addWidget(self.add_particles_btn)
        
        self.clear_particles_btn = QPushButton("Clear Particles")
        self.clear_particles_btn.clicked.connect(self.clear_particles)
        controls_layout.addWidget(self.clear_particles_btn)
        
        controls_layout.addWidget(QLabel("Particle Count:"))
        self.particle_count_spin = QSpinBox()
        self.particle_count_spin.setRange(10, 500)
        self.particle_count_spin.setValue(50)
        controls_layout.addWidget(self.particle_count_spin)
        
        controls_layout.addWidget(QLabel("Speed Factor:"))
        self.speed_slider = QSlider(Qt.Horizontal) # type: ignore
        self.speed_slider.setRange(10, 200)
        self.speed_slider.setValue(100)
        controls_layout.addWidget(self.speed_slider)
        
        layout.addLayout(controls_layout)
        
        self.plotter = QtInteractor(self)
        self.plotter.reset_camera_clipping_range_mode = 'never'
        layout.addWidget(self.plotter.interactor)
        
        self.initialize()
        
    def initialize(self):
        from burger_vortex import BurgersVortex
        self.bv = BurgersVortex()
        
        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.plotter.camera_position = [(15, 15, 15), (0, 0, 0), (0, 0, 1)]
        self.plotter.reset_camera()
        self.original_camera_position = self.plotter.camera_position
        
        self.particles = None
        self.particle_positions = np.empty((0, 3))
        self.particle_colors = np.empty((0, 4))
        self.particle_mesh = None
        
        self.add_random_particles()
    
    def add_random_particles(self):
        count = self.particle_count_spin.value()
        
        new_positions = []
        for _ in range(count):
            r = 1.0 + 2.0 * np.random.random()
            theta = 2 * np.pi * np.random.random()
            phi = np.pi * np.random.random()
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) * 2
            
            new_positions.append([x, y, z])
        
        new_positions = np.array(new_positions)
        if self.particle_positions.size == 0:
            self.particle_positions = new_positions
        else:
            self.particle_positions = np.vstack([self.particle_positions, new_positions])
        
        radii = np.linalg.norm(new_positions, axis=1)
        norm_radii = (radii - np.min(radii)) / (np.max(radii) - np.min(radii) + 1e-10)
        
        new_colors = np.zeros((len(new_positions), 4))
        new_colors[:, 0] = norm_radii
        new_colors[:, 2] = 1 - norm_radii
        new_colors[:, 1] = 0.2 * np.ones_like(norm_radii)
        new_colors[:, 3] = 0.8 * np.ones_like(norm_radii)
        
        if self.particle_colors.size == 0:
            self.particle_colors = new_colors
        else:
            self.particle_colors = np.vstack([self.particle_colors, new_colors])
        
        self._update_particle_mesh()
    
    def clear_particles(self):
        self.particle_positions = np.empty((0, 3))
        self.particle_colors = np.empty((0, 4))
        if self.particle_mesh:
            self.plotter.remove_actor(self.particle_mesh, render=False)
            self.particle_mesh = None
        self.plotter.render()
    
    def _update_particle_mesh(self):
        if self.particle_positions.size == 0:
            return
            
        current_camera = self.plotter.camera_position
            
        point_cloud = pv.PolyData(self.particle_positions)
        point_cloud['colors'] = self.particle_colors * 255
        
        if self.particle_mesh:
            self.plotter.remove_actor(self.particle_mesh, render=False)
        
        self.particle_mesh = self.plotter.add_mesh(
            point_cloud,
            render_points_as_spheres=True,
            point_size=10,
            rgb=True,
            scalars='colors',
            reset_camera=False,
            render=False
        )
        
        self.plotter.camera_position = current_camera
        self.plotter.render()
    
    def update_kinematic(self, time_elapsed):
        if self.particle_positions.size == 0:
            return
            
        speed_factor = self.speed_slider.value() / 100.0
        delta_t = 0.05 * speed_factor
        
        for i in range(len(self.particle_positions)):
            pos = self.particle_positions[i]
            velocity = self.bv.velocity2(pos[0], pos[1], pos[2])
            self.particle_positions[i] += velocity * delta_t
        
        distances = np.linalg.norm(self.particle_positions, axis=1)
        mask = distances < 10.0
        
        if not np.all(mask):
            self.particle_positions = self.particle_positions[mask]
            self.particle_colors = self.particle_colors[mask]
            
            particles_to_add = np.sum(~mask)
            if particles_to_add > 0:
                new_positions = []
                for _ in range(particles_to_add):
                    r = 3.0
                    theta = 2 * np.pi * np.random.random()
                    z = 4.0 * (np.random.random() - 0.5)
                    
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    
                    new_positions.append([x, y, z])
                
                new_positions = np.array(new_positions)
                self.particle_positions = np.vstack([self.particle_positions, new_positions])
                
                new_colors = np.zeros((particles_to_add, 4))
                new_colors[:, 0] = 0.8
                new_colors[:, 1] = 0.2
                new_colors[:, 2] = 0.2
                new_colors[:, 3] = 0.8
                
                self.particle_colors = np.vstack([self.particle_colors, new_colors])
        
        self._update_particle_mesh()

class LiutexTab(QWidget):
    def __init__(self, max_x_value, num_of_points):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.max_x_value = max_x_value
        self.num_of_points = num_of_points

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Coloring Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Q", "delta", "lambda_ci", "lambda_2", "liutex", "velocity_magnitude"])
        self.method_combo.setCurrentText("Q")
        self.method_combo.currentTextChanged.connect(self.update_visualization)
        method_layout.addWidget(self.method_combo)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.update_visualization)
        method_layout.addWidget(refresh_button)
        
        layout.addLayout(method_layout)

        self.plotter = QtInteractor(self)
        self.plotter.add_axes()
        self.plotter.view_isometric()
        
        layout.addWidget(self.plotter.interactor)

        self.bv = bv.BurgersVortex()

        self.compute_and_plot_streamlines()
        
    def update_visualization(self):
        self.plotter.clear()
        self.compute_and_plot_streamlines()

    def compute_and_plot_streamlines(self):
        max_range = min(2.0, self.max_x_value)
        
        method = self.method_combo.currentText() if hasattr(self, 'method_combo') else "Q"
        
        num_points = max(self.num_of_points, 30)
        x = np.linspace(-max_range, max_range, num_points)
        y = np.linspace(-max_range, max_range, num_points)
        z = np.linspace(-max_range, max_range, num_points)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        vectors = np.zeros((X.size, 3))
        scalars = np.zeros(X.size)
        
        for i in range(X.size):
            xi, yi, zi = X.flat[i], Y.flat[i], Z.flat[i]
            vectors[i] = self.bv.velocity2(xi, yi, zi)
            scalars[i] = self.bv.return_coloring(xi, yi, zi, method)

        grid = pv.StructuredGrid()
        grid.points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        grid.dimensions = X.shape
        grid['vectors'] = vectors
        grid[method] = scalars
        
        start_points = []
        for radius in [0.5, 1.0, 1.5]:
            for angle in np.linspace(0, 2*np.pi, 8):
                for height in [-0.5, 0.0, 0.5]:
                    start_points.append([radius * np.cos(angle), radius * np.sin(angle), height])
        
        start_points = np.array(start_points)
        
        streams = grid.streamlines_from_source(
            source=pv.PolyData(start_points),
            vectors='vectors',
            max_steps=500,
            integration_direction='forward'
        )
        
        stream_scalars = np.zeros(streams.n_points)
        for i, point in enumerate(streams.points):
            x, y, z = point
            stream_scalars[i] = self.bv.return_coloring(x, y, z, method)
        
        finite_values = stream_scalars[np.isfinite(stream_scalars)]
        p5 = np.percentile(finite_values, 5)
        p95 = np.percentile(finite_values, 95)
        stream_scalars = np.clip(stream_scalars, p5, p95)
        stream_scalars = (stream_scalars - p5) / (p95 - p5)
        
        streams[method + '_normalized'] = stream_scalars
        
        tubes = streams.tube(radius=0.02)
        self.plotter.add_mesh(
            tubes, 
            scalars=method + '_normalized',
            cmap='plasma',
            show_scalar_bar=True,
            scalar_bar_args={'title': f'{method} Criterion'}
        )
        
        self.plotter.reset_camera()

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

        maxInputValue = 10000.00

        def sign(value):
            return (value > 0) - (value < 0)

        def is_float(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        def check_value(obj):
            text = obj.text()
            if text and is_float(text):
                value = float(text)
                if abs(value) > maxInputValue:
                    obj.setText(f"{sign(value) * maxInputValue:.2f}")
        
        projectile_layout = QVBoxLayout()
        projectile_layout.addWidget(QLabel("Projectile details"))
        
        pos_input = QHBoxLayout()
        pos_input.addWidget(QLabel("Projectile starting postition (x, y, z)"))
        positionVec = [0.0, 3.0, 30.0]
        for i in range(3):
            le = QLineEdit(str(positionVec[i]))
            le.textChanged.connect(lambda _t, w=le: check_value(w))
            pos_input.addWidget(le)
            
        vel_input = QHBoxLayout()
        vel_input.addWidget(QLabel("Projectile starting velocity (x, y, z)"))
        velosityVec = [0.0, 0.0, 0.0]
        for i in range(3):
            le = QLineEdit(str(velosityVec[i]))
            le.textChanged.connect(lambda _t, w=le: check_value(w))
            vel_input.addWidget(le)
            
        projectile_layout.addLayout(pos_input)
        projectile_layout.addLayout(vel_input)
        
        mass_input_layout = QHBoxLayout()
        mass_input_layout.addWidget(QLabel("Projectile mass in kg"))
        mass_input = QLineEdit("20.0")
        mass_input.setObjectName("mass")
        mass_input.setValidator(QDoubleValidator(-maxInputValue, maxInputValue, 2))
        mass_input.textChanged.connect(lambda: check_value(mass_input))
        mass_input_layout.addWidget(mass_input)
       
        radius_input_layout = QHBoxLayout()
        radius_input_layout.addWidget(QLabel("Projectile radius in m"))
        radius_input = QLineEdit("3.0")
        radius_input.setObjectName("radius")
        radius_input.setValidator(QDoubleValidator(-maxInputValue, maxInputValue, 2))
        radius_input.textChanged.connect(lambda: check_value(radius_input))
        radius_input_layout.addWidget(radius_input)
        
        projectile_layout.addLayout(mass_input_layout)
        projectile_layout.addLayout(radius_input_layout)
        
        tornado_layout = QVBoxLayout()
        tornado_layout.addWidget(QLabel("Tornado details"))
        
        alpha_input_layout = QHBoxLayout()
        alpha_input_layout.addWidget(QLabel("Alpha"))
        alpha_input = QLineEdit("0.1")
        alpha_input.setObjectName("Alpha")
        alpha_input.setValidator(QDoubleValidator(-maxInputValue, maxInputValue, 2))
        alpha_input.textChanged.connect(lambda: check_value(alpha_input))
        alpha_input_layout.addWidget(alpha_input) 
        tornado_layout.addLayout(alpha_input_layout)
        
        nu_input_layout = QHBoxLayout()
        nu_input_layout.addWidget(QLabel("Nu"))
        nu_input = QLineEdit("1.5e-5")
        nu_input.setObjectName("Nu")
        nu_input.setValidator(QDoubleValidator(-maxInputValue, maxInputValue, 2))
        nu_input.textChanged.connect(lambda: check_value(nu_input))
        nu_input_layout.addWidget(nu_input)
        tornado_layout.addLayout(nu_input_layout)
        
        gamma_input_layout = QHBoxLayout()
        gamma_input_layout.addWidget(QLabel("Gamma"))
        gamma_input = QLineEdit("2000")
        gamma_input.setObjectName("Gamma")
        gamma_input.setValidator(QDoubleValidator(-maxInputValue, maxInputValue, 2))
        gamma_input.textChanged.connect(lambda: check_value(gamma_input))
        gamma_input_layout.addWidget(gamma_input)
        tornado_layout.addLayout(gamma_input_layout)
        
        self.launch_button = QPushButton("Launch Projectile")
        self.launch_button.clicked.connect(self.launch_projectile)

        layout.addLayout(projectile_layout)
        layout.addLayout(tornado_layout)
        layout.addWidget(self.launch_button)

    def launch_projectile(self):
        global user_data_settings

        mainWindow = self.parent()
        while mainWindow and not isinstance(mainWindow, QMainWindow):
            mainWindow = mainWindow.parent()

        if mainWindow is None:
            return
            
        mainWindow.stop_update()
        user_data_settings = {
            "position": [float(x.text()) for x in self.findChildren(QLineEdit)[:3]],
            "velocity": [float(x.text()) for x in self.findChildren(QLineEdit)[3:6]],
            "mass": float(self.findChild(QLineEdit, "mass").text()),
            "radius": float(self.findChild(QLineEdit, "radius").text()),
            "alpha": float(self.findChild(QLineEdit, "Alpha").text()),
            "nu": float(self.findChild(QLineEdit, "Nu").text()),
            "gamma": float(self.findChild(QLineEdit, "Gamma").text())
        }

        print(user_data_settings)
        mainWindow.update_user_data_settings()
        
        if not hasattr(mainWindow, 'timer'):
            mainWindow.timer = QTimer()
            mainWindow.timer.timeout.connect(mainWindow.update_all)
            mainWindow.timer.start(16)
            mainWindow.time = 0.0

        if not hasattr(mainWindow, 'kinematic_timer'):
            mainWindow.kinematic_timer = QTimer()
            mainWindow.kinematic_timer.timeout.connect(mainWindow.update_kinematic)
            mainWindow.kinematic_timer.start(16)
            mainWindow.kinematicTime = 0.0

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

        self.kinematic_tab = particleSimulationTab()
        self.sim_tab = TornadoSimulationTab(self.tornado)
        self.plot_tab = PlottingTab(self.tornado, ["vx", "vy", "vz"], lambda: self.tornado.projectile.velocity)
        self.plot_tab_acc = PlottingTab(self.tornado, ["ax", "ay", "az"], lambda: self.tornado.projectile.acceleration)

        if show_liutex_tab:
            assert liutex_settings_selected is not None, "Liutex settings must be selected before showing the tab."
            self.liutex_tab = LiutexTab(liutex_settings[liutex_settings_selected][0], liutex_settings[liutex_settings_selected][1])

        self.tabs.addTab(self.sim_tab, "Tornado Simulation")
        self.tabs.addTab(self.plot_tab, "Tornado Plotting Vel")
        self.tabs.addTab(self.plot_tab_acc, "Tornado Plotting Acc")
        self.tabs.addTab(self.kinematic_tab, "Particle Tab")
        
        if show_liutex_tab:
            self.tabs.addTab(self.liutex_tab, "3d Mesh")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.control_panel = ControlPanel(self.tornado)
        main_layout.addWidget(self.control_panel)
        
        self.setCentralWidget(main_widget)
    
    def update_user_data_settings(self):
        global user_data_settings
        self.tornado = Tornado(user_data_settings["position"],
                                user_data_settings["velocity"],
                                user_data_settings["mass"],
                                user_data_settings["radius"],
                                user_data_settings["alpha"],
                                user_data_settings["nu"],
                                user_data_settings["gamma"])
        self.sim_tab.tornado = self.tornado
        self.plot_tab.tornado = self.tornado
        self.plot_tab_acc.tornado = self.tornado
    
    def on_tab_changed(self):
        tabText = self.tabs.tabText(self.tabs.currentIndex())
        if tabText == "3d Mesh":
            self.control_panel.setVisible(False)
        else:
            self.control_panel.setVisible(True)
    
    def update_tornado(self):
        self.tornado = Tornado()
        self.sim_tab.tornado = self.tornado
        self.plot_tab.tornado = self.tornado
        self.plot_tab_acc.tornado = self.tornado

    def update_all(self):
        if self.tornado.projectile.position[2] > 0 and self.tornado.projectile.position[2] < 500:
            delta_time = 0.01
            self.sim_tab.update_simulation(delta_time)
            self.time += delta_time
            self.plot_tab.update_plots(self.time)
            self.plot_tab_acc.update_plots(self.time)
        else:
            self.timer.stop()
    
    def stop_update(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer

        if hasattr(self, 'kinematic_timer'):
            self.kinematic_timer.stop()
            self.kinematic_timer.deleteLater()
            del self.kinematic_timer

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
    
    def update_kinematic(self):
        delta_time = 0.01
        self.kinematicTime += delta_time
        self.kinematic_tab.update_kinematic(self.kinematicTime)

class PopUpWindow(QMainWindow):
    ok_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3d Mesh settings")
        self.setGeometry(100, 100, 300, 100)
        
        layout = QVBoxLayout()
        self.checkbox = QCheckBox("Show 3d Mesh Tab")
        self.checkbox.setChecked(False)
        self.checkbox.stateChanged.connect(self.toggle_listbox_visibility)
        layout.addWidget(self.checkbox)

        self.listbox = QListWidget()
        self.listbox.addItems(liutex_settings.keys())
        self.listbox.setVisible(False)
        layout.addWidget(self.listbox)
        
        button = QPushButton("OK")
        button.clicked.connect(self.on_ok_clicked)
        layout.addWidget(button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def on_ok_clicked(self):
        global show_liutex_tab, liutex_settings_selected
        show_liutex_tab = self.checkbox.isChecked()
        liutex_settings_selected = self.listbox.currentItem().text() if self.listbox.currentItem() else None # type: ignore
        self.close()
        self.ok_clicked.emit()
    
    def toggle_listbox_visibility(self):
        self.listbox.setVisible(self.checkbox.isChecked())

def main():
    app = QApplication(sys.argv)
    popup_window = PopUpWindow()
    popup_window.show()

    def open_main_window():
        main_window = MainWindow()
        main_window.show()
    
    popup_window.ok_clicked.connect(open_main_window)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()