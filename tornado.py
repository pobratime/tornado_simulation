import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QGridLayout, QPushButton, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QListWidget, QComboBox
from PyQt5.QtCore import QTimer, pyqtSignal
from pyvistaqt import QtInteractor
import pyqtgraph.opengl as gl
import pyqtgraph as pq
import pyvista as pv
from tqdm import tqdm
import burger_vortex as bv

show_liutex_tab = False
liutex_settings_selected = None
liutex_settings = {
    "Low": (4, 25),
    "Medium": (4, 100),
    "High": (4, 500),
    "Very High": (4, 1000)
}

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
        self.mass = 10
        self.radius = 1
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
        
        self.nu = 0
        self.gamma = 0
        self.alpha = 0
        
        self.bvortex = bv.BurgersVortex()
        
        self.inflow_angle = np.deg2rad(0)
        self.K = 0.5

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
        return self.bvortex.velocity(x, y, z)
         
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
        self.view.opts['distance'] = 5
        self.number_of_particles = 500
        self.max_radius = 2.0
        self.spawn_radius = 1.8
        self.particle_lifetime = 0.0
        
        self.positions = self.initialize_particles()
        self.particle_ages = np.zeros(self.number_of_particles)  
        self.max_age = 15.0
        
        self.vortex = bv.BurgersVortex()
        
        colors = self.get_particle_colors()
        self.scatter = gl.GLScatterPlotItem(pos=self.positions, size=4, color=colors, pxMode=True)
        self.view.addItem(self.scatter)
        layout.addWidget(self.view)
    
    def initialize_particles(self):
        positions = np.zeros((self.number_of_particles, 3))
        for i in range(self.number_of_particles):
            radius = np.random.uniform(0.5, self.spawn_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            height = np.random.uniform(-1.0, 1.0)
            
            positions[i] = [
                radius * np.cos(angle),
                radius * np.sin(angle), 
                height
            ]
        return positions
    
    def spawn_new_particle(self, index):
        radius = np.random.uniform(self.spawn_radius * 0.9, self.spawn_radius)
        angle = np.random.uniform(0, 2 * np.pi)
        height = np.random.uniform(-1.0, 1.0)
        
        self.positions[index] = [
            radius * np.cos(angle),
            radius * np.sin(angle),
            height
        ]
        self.particle_ages[index] = 0.0
    
    def get_particle_colors(self):
        colors = np.zeros((self.number_of_particles, 4))
        for i in range(self.number_of_particles):
            age_factor = min(self.particle_ages[i] / self.max_age, 1.0)
            brightness = 1.0 - age_factor * 0.7
            
            colors[i] = [0.2 * brightness, 0.8 * brightness, 1.0 * brightness, 1.0]
        return colors

    def update_kinematic(self, delta_time):
        particles_to_respawn = []
        
        for i in range(self.number_of_particles):
            self.particle_ages[i] += delta_time
            
            vel = self.vortex.velocity(*self.positions[i])
            self.positions[i] += vel * delta_time
            
            r = np.linalg.norm(self.positions[i][:2])
            z = self.positions[i][2]
            
            if (r > self.max_radius or 
                abs(z) > 2.0 or 
                self.particle_ages[i] > self.max_age or
                r < 0.05):
                particles_to_respawn.append(i)
        
        for i in particles_to_respawn:
            self.spawn_new_particle(i)
        
        colors = self.get_particle_colors()
        
        self.scatter.setData(pos=self.positions, color=colors)

class LiutexTab(QWidget):
    def __init__(self, max_x_value, num_of_points):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.max_x_value = max_x_value
        self.num_of_points = num_of_points

        # Add method selection
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
        layout.addWidget(self.plotter.interactor)

        self.bv = bv.BurgersVortex()

        self.compute_and_plot_streamlines()
        
    def update_visualization(self):
        self.plotter.clear()
        self.compute_and_plot_streamlines()

    def compute_and_plot_streamlines(self):
        # Use smaller range for better visualization of vortex structures
        max_range = min(2.0, self.max_x_value)  # Limit to 2.0 for better vortex capture
        
        # Get selected method
        method = self.method_combo.currentText() if hasattr(self, 'method_combo') else "Q"
        
        # Create a finer grid for better resolution
        num_points = max(self.num_of_points, 30)  # Ensure minimum resolution
        x = np.linspace(-max_range, max_range, num_points)
        y = np.linspace(-max_range, max_range, num_points)
        z = np.linspace(-max_range, max_range, num_points)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        vectors = np.zeros((X.size, 3))
        scalars = np.zeros(X.size)

        # Compute vectors and scalars for the grid
        for i in range(X.size):
            xi, yi, zi = X.flat[i], Y.flat[i], Z.flat[i]
            vectors[i] = self.bv.velocity2(xi, yi, zi)
            scalars[i] = self.bv.return_coloring(xi, yi, zi, method)

        # Create the grid
        grid = pv.StructuredGrid()
        grid.points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        grid.dimensions = X.shape
        grid['vectors'] = vectors
        grid[method] = scalars
        
        # Create multiple streamlines from different starting points around the vortex
        start_points = []
        # Create a circle of starting points at different radii and heights
        for radius in [0.5, 1.0, 1.5]:
            for angle in np.linspace(0, 2*np.pi, 8):
                for height in [-0.5, 0.0, 0.5]:
                    x_start = radius * np.cos(angle)
                    y_start = radius * np.sin(angle)
                    start_points.append([x_start, y_start, height])
        
        start_points = np.array(start_points)
        
        try:
            # Create streamlines with custom starting points
            streams = grid.streamlines_from_source(
                source=pv.PolyData(start_points),
                vectors='vectors',
                max_steps=500,
                integration_direction='forward'
            )
            
            if streams.n_points > 0:
                # Compute scalars along the streamlines
                stream_scalars = np.zeros(streams.n_points)
                points = streams.points
                
                for i, point in enumerate(points):
                    x, y, z = point
                    stream_scalars[i] = self.bv.return_coloring(x, y, z, method)
                
                # Normalize for better color distribution
                if len(stream_scalars) > 0:
                    finite_values = stream_scalars[np.isfinite(stream_scalars)]
                    if len(finite_values) > 0:
                        p5 = np.percentile(finite_values, 5)
                        p95 = np.percentile(finite_values, 95)
                        stream_scalars = np.clip(stream_scalars, p5, p95)
                        if p95 > p5:
                            stream_scalars = (stream_scalars - p5) / (p95 - p5)
                
                streams[method + '_normalized'] = stream_scalars
                
                # Create tubes from streamlines
                tubes = streams.tube(radius=0.02)
                
                # Add to plotter
                self.plotter.add_mesh(
                    tubes, 
                    scalars=method + '_normalized',
                    cmap='plasma',
                    show_scalar_bar=True,
                    scalar_bar_args={'title': f'{method} Criterion'}
                )
            else:
                # Fallback: show grid points
                finite_mask = np.isfinite(scalars)
                if np.any(finite_mask):
                    finite_scalars = scalars[finite_mask]
                    p5 = np.percentile(finite_scalars, 5)
                    p95 = np.percentile(finite_scalars, 95)
                    scalars_norm = np.clip(scalars, p5, p95)
                    if p95 > p5:
                        scalars_norm = (scalars_norm - p5) / (p95 - p5)
                    grid[method + '_normalized'] = scalars_norm
                    self.plotter.add_mesh(
                        grid, 
                        scalars=method + '_normalized',
                        cmap='plasma',
                        show_scalar_bar=True,
                        style='points',
                        point_size=5
                    )
                
        except Exception as e:
            print(f"Streamline error: {e}")
            # Fallback: show just the grid points with computed scalars
            finite_mask = np.isfinite(scalars)
            if np.any(finite_mask):
                finite_scalars = scalars[finite_mask]
                p5 = np.percentile(finite_scalars, 5)
                p95 = np.percentile(finite_scalars, 95)
                scalars_norm = np.clip(scalars, p5, p95)
                if p95 > p5:
                    scalars_norm = (scalars_norm - p5) / (p95 - p5)
                grid[method + '_normalized'] = scalars_norm
                self.plotter.add_mesh(
                    grid, 
                    scalars=method + '_normalized',
                    cmap='plasma',
                    show_scalar_bar=True,
                    style='points',
                    point_size=3
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

        if mainWindow is None:
            return
            
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

        if show_liutex_tab:
            assert liutex_settings_selected is not None, "Liutex settings must be selected before showing the tab."
            self.liutex_tab = LiutexTab(liutex_settings[liutex_settings_selected][0], liutex_settings[liutex_settings_selected][1])

        self.tabs.addTab(self.sim_tab, "Tornado Simulation")
        self.tabs.addTab(self.kinematic_tab, "KinematicTab")
        self.tabs.addTab(self.plot_tab, "Tornado Plotting Vel")
        self.tabs.addTab(self.plot_tab_acc, "Tornado Plotting Acc")

        if show_liutex_tab:
            self.tabs.addTab(self.liutex_tab, "Liutex")

        self.control_panel = ControlPanel(self.tornado)
        main_layout.addWidget(self.control_panel)
        
        self.setCentralWidget(main_widget)
    
    def update_tornado(self):
        self.tornado = Tornado()
        self.sim_tab.tornado = self.tornado
        self.plot_tab.tornado = self.tornado
        self.plot_tab_acc.tornado = self.tornado

    def update_all(self):
        if self.tornado.projectile.position[2] > 0 or self.tornado.projectile.position[2] < 500:
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

class PopUpWindow(QMainWindow):
    ok_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liutex settings")
        self.setGeometry(100, 100, 300, 100)
        
        layout = QVBoxLayout()
        self.checkbox = QCheckBox("Show Liutex Tab")
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
        liutex_settings_selected = self.listbox.currentItem().text() if self.listbox.currentItem() else None
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