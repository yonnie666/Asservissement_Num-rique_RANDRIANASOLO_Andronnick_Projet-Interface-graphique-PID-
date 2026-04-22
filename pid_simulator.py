import tkinter as tk
from tkinter import ttk
import numpy as np
from collections import deque

class PIDController:
    def __init__(self):
        self.kp = 2.0
        self.ki = 0.5
        self.kd = 0.1
        self.setpoint = 100
        self.prev_error = 0
        self.integral = 0
        self.dt = 0.1
        
    def compute(self, measurement):
        error = self.setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return max(0, min(200, output))

class ProcessSimulator:
    def __init__(self):
        self.value = 0
        self.gain = 1.0
        self.time_constant = 5.0
        
    def update(self, control):
        dt = 0.1
        alpha = dt / self.time_constant
        self.value = self.value * (1 - alpha) + self.gain * control * alpha
        noise = np.random.normal(0, 0.5)
        return max(0, self.value + noise)

class PIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulateur PID")
        self.root.geometry("900x700")
        
        self.running = False
        self.time_data = deque(maxlen=200)
        self.process_data = deque(maxlen=200)
        self.setpoint_data = deque(maxlen=200)
        self.control_data = deque(maxlen=200)
        self.time_counter = 0
        
        self.pid = PIDController()
        self.process = ProcessSimulator()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panneau de contrôle (gauche)
        control_frame = ttk.LabelFrame(main_frame, text="Paramètres PID", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Kp
        ttk.Label(control_frame, text="Kp (Proportionnel):").pack(anchor=tk.W, pady=(0, 5))
        self.kp_var = tk.DoubleVar(value=2.0)
        kp_scale = ttk.Scale(control_frame, from_=0, to=10, variable=self.kp_var, orient=tk.HORIZONTAL, length=250)
        kp_scale.pack(fill=tk.X, pady=(0, 5))
        self.kp_label = ttk.Label(control_frame, text="2.00")
        self.kp_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Ki
        ttk.Label(control_frame, text="Ki (Intégral):").pack(anchor=tk.W, pady=(0, 5))
        self.ki_var = tk.DoubleVar(value=0.5)
        ki_scale = ttk.Scale(control_frame, from_=0, to=5, variable=self.ki_var, orient=tk.HORIZONTAL, length=250)
        ki_scale.pack(fill=tk.X, pady=(0, 5))
        self.ki_label = ttk.Label(control_frame, text="0.50")
        self.ki_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Kd
        ttk.Label(control_frame, text="Kd (Dérivé):").pack(anchor=tk.W, pady=(0, 5))
        self.kd_var = tk.DoubleVar(value=0.1)
        kd_scale = ttk.Scale(control_frame, from_=0, to=2, variable=self.kd_var, orient=tk.HORIZONTAL, length=250)
        kd_scale.pack(fill=tk.X, pady=(0, 5))
        self.kd_label = ttk.Label(control_frame, text="0.10")
        self.kd_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Mise à jour labels
        self.kp_var.trace('w', lambda *a: self.kp_label.config(text=f"{self.kp_var.get():.2f}"))
        self.ki_var.trace('w', lambda *a: self.ki_label.config(text=f"{self.ki_var.get():.2f}"))
        self.kd_var.trace('w', lambda *a: self.kd_label.config(text=f"{self.kd_var.get():.2f}"))
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Consigne
        ttk.Label(control_frame, text="Consigne:").pack(anchor=tk.W, pady=(0, 5))
        setpoint_frame = ttk.Frame(control_frame)
        setpoint_frame.pack(fill=tk.X, pady=(0, 15))
        self.setpoint_var = tk.DoubleVar(value=100)
        ttk.Entry(setpoint_frame, textvariable=self.setpoint_var, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(setpoint_frame, text="Appliquer", command=self.update_setpoint).pack(side=tk.LEFT)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Boutons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Démarrer", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏸ Pause", command=self.stop, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="⟳ Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        
        # Zone d'affichage (droite)
        display_frame = ttk.LabelFrame(main_frame, text="Visualisation", padding="10")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas pour dessiner
        self.canvas = tk.Canvas(display_frame, bg='white', height=350)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Informations texte
        self.info_text = tk.Text(display_frame, height=12, font=('Courier', 10))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
    def update_setpoint(self):
        self.pid.setpoint = self.setpoint_var.get()
        
    def start(self):
        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.simulation_loop()
        
    def stop(self):
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
    def reset(self):
        self.running = False
        self.time_data.clear()
        self.process_data.clear()
        self.setpoint_data.clear()
        self.control_data.clear()
        self.time_counter = 0
        self.process.value = 0
        self.pid.prev_error = 0
        self.pid.integral = 0
        self.update_display()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
    def simulation_loop(self):
        if not self.running:
            return
            
        # Mise à jour paramètres
        self.pid.kp = self.kp_var.get()
        self.pid.ki = self.ki_var.get()
        self.pid.kd = self.kd_var.get()
        
        # Calcul
        current = self.process_data[-1] if self.process_data else 0
        control = self.pid.compute(current)
        measurement = self.process.update(control)
        
        # Stockage
        self.time_counter += 0.1
        self.time_data.append(self.time_counter)
        self.setpoint_data.append(self.pid.setpoint)
        self.process_data.append(measurement)
        self.control_data.append(control)
        
        # Affichage
        self.update_display()
        
        # Boucle
        self.root.after(100, self.simulation_loop)
        
    def update_display(self):
        # Mise à jour du canvas
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width > 10 and len(self.process_data) > 1:
            # Trouver les valeurs max
            all_values = list(self.setpoint_data) + list(self.process_data)
            max_val = max(all_values) if all_values else 150
            min_val = min(0, min(all_values)) if all_values else 0
            range_val = max_val - min_val if max_val != min_val else 150
            
            # Dessiner la grille
            for i in range(5):
                y = height - (i * height / 4)
                self.canvas.create_line(0, y, width, y, fill='#e0e0e0', dash=(2, 2))
                value = min_val + (i * range_val / 4)
                self.canvas.create_text(5, y - 5, text=f"{value:.0f}", fill='#666', font=('Arial', 8), anchor=tk.W)
            
            # Dessiner la consigne
            setpoint_y = height - ((self.pid.setpoint - min_val) / range_val) * height
            if 0 <= setpoint_y <= height:
                self.canvas.create_line(0, setpoint_y, width, setpoint_y, fill='green', dash=(5, 5), width=2)
            
            # Dessiner la mesure
            points = []
            for i, val in enumerate(self.process_data):
                x = (i / max(1, len(self.process_data))) * width
                y = height - ((val - min_val) / range_val) * height
                if 0 <= y <= height:
                    points.append((x, y))
            
            if len(points) > 1:
                for i in range(len(points)-1):
                    if points[i][1] > 0 and points[i+1][1] > 0:
                        self.canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], fill='blue', width=2)
        
        # Mise à jour des informations
        self.info_text.delete(1.0, tk.END)
        current = self.process_data[-1] if self.process_data else 0
        error = abs(self.pid.setpoint - current)
        
        info = f"""
╔════════════════════════════════════════════╗
║              ÉTAT DU SYSTÈME               ║
╠════════════════════════════════════════════╣
║ Temps écoulé     : {self.time_counter:>6.1f} s
║                                            ║
║ Paramètres PID :                           ║
║   Kp = {self.pid.kp:>6.2f}
║   Ki = {self.pid.ki:>6.2f}
║   Kd = {self.pid.kd:>6.2f}
║                                            ║
║ Performances :                             ║
║   Consigne = {self.pid.setpoint:>6.1f}
║   Mesure   = {current:>6.1f}
║   Erreur   = {error:>6.1f}
║   Commande = {self.control_data[-1] if self.control_data else 0:>6.1f}
║                                            ║
╚════════════════════════════════════════════╝
        """
        self.info_text.insert(1.0, info)

if __name__ == "__main__":
    root = tk.Tk()
    app = PIDApp(root)
    root.mainloop()