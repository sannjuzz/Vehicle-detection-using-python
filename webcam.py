
import cv2
import easyocr
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import webbrowser
import json
import os
from datetime import datetime
from tkinter import font as tkfont

# ================= CONFIGURATION =================
CONFIG = {
    "ADMIN_CREDENTIALS": {"username": "admin", "password": "admin123"},
    "MODEL_CONFIDENCE": 0.25,
    "OCR_CONFIDENCE": 0.25,
    "DATABASE_FILE": "vehicle_database.json",
    "THEME": {
        "bg_primary": "#1a1a2e",
        "bg_secondary": "#16213e",
        "bg_tertiary": "#0f3460",
        "accent": "#e94560",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "info": "#2196F3",
        "text_primary": "#FFFFFF",
        "text_secondary": "#CCCCCC"
    }
}

# ================= VEHICLE DATABASE =================
class VehicleDatabase:
    def __init__(self, filename=CONFIG["DATABASE_FILE"]):
        self.filename = filename
        self.db = self.load_database()
    
    def load_database(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                return {}
        return {}
    
    def save_database(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.db, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_vehicle(self, plate, from_place, to_place):
        self.db[plate] = {
            "from": from_place,
            "to": to_place,
            "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return self.save_database()
    
    def get_vehicle(self, plate):
        return self.db.get(plate)
    
    def remove_vehicle(self, plate):
        if plate in self.db:
            del self.db[plate]
            return self.save_database()
        return False

# ================= HELPER FUNCTIONS =================
def normalize_plate(text):
    """Normalize license plate text"""
    if not text:
        return ""
    return ''.join(c for c in text.upper() if c.isalnum())

def get_ocr_text(image, bbox, reader, conf_thresh=CONFIG["OCR_CONFIDENCE"]):
    """Extract text from bounding box using OCR"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return ""
    
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    
    # Enhance image for better OCR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    results = reader.readtext(gray)
    
    best_text, best_conf = "", conf_thresh
    for bbox, text, conf in results:
        if conf > best_conf:
            best_text, best_conf = text, conf
    
    return best_text.strip()

def create_gradient(width, height, color1, color2):
    """Create gradient background"""
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (width, height), color1)
    draw = ImageDraw.Draw(image)
    
    for i in range(height):
        ratio = i / height
        r = int(color1[0] + (color2[0] - color1[0]) * ratio)
        g = int(color1[1] + (color2[1] - color1[1]) * ratio)
        b = int(color1[2] + (color2[2] - color1[2]) * ratio)
        draw.line([(0, i), (width, i)], fill=(r, g, b))
    
    return ImageTk.PhotoImage(image)

# ================= MAIN APPLICATION =================
class NumberPlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Intelligent Vehicle Recognition System")
        self.root.geometry("1280x820")
        self.root.minsize(1200, 780)
        
        # Initialize variables
        self.running = False
        self.cap = None
        self.last_map_opened = ""
        self.vehicle_db = VehicleDatabase()
        self.theme = CONFIG["THEME"]
        
        # Configure root window
        self.root.configure(bg=self.theme["bg_primary"])
        
        # Initialize models (lazy loading)
        self.model = None
        self.reader = None
        self.model_loaded = False
        
        # Build UI
        self.setup_fonts()
        self.build_ui()
        
        # Center window
        self.center_window()
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_fonts(self):
        """Setup custom fonts"""
        self.title_font = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.header_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.body_font = tkfont.Font(family="Segoe UI", size=11)
        self.mono_font = tkfont.Font(family="Consolas", size=10)

    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def build_ui(self):
        """Build the main user interface"""
        # Main container
        main_container = tk.Frame(self.root, bg=self.theme["bg_primary"])
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_container, bg=self.theme["bg_primary"])
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="INTELLIGENT VEHICLE RECOGNITION SYSTEM",
            font=self.title_font,
            fg=self.theme["accent"],
            bg=self.theme["bg_primary"]
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Real-time License Plate Detection & Analysis",
            font=self.body_font,
            fg=self.theme["text_secondary"],
            bg=self.theme["bg_primary"]
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Control Panel
        control_frame = tk.Frame(
            main_container,
            bg=self.theme["bg_secondary"],
            relief="flat",
            bd=0
        )
        control_frame.pack(fill="x", pady=(0, 20))
        
        # Control buttons
        btn_style = {
            "padding": (20, 10),
            "style": "Accent.TButton"
        }
        
        self.start_btn = ttk.Button(
            control_frame,
            text="▶ START DETECTION",
            command=self.start_camera,
            **btn_style
        )
        self.start_btn.pack(side="left", padx=(20, 10), pady=15)
        
        self.stop_btn = ttk.Button(
            control_frame,
            text="⏹ STOP DETECTION",
            command=self.stop_camera,
            state="disabled",
            **btn_style
        )
        self.stop_btn.pack(side="left", padx=10, pady=15)
        
        # Admin button on right
        self.admin_btn = ttk.Button(
            control_frame,
            text="🔐 ADMIN PANEL",
            command=self.admin_login,
            style="Admin.TButton"
        )
        self.admin_btn.pack(side="right", padx=(10, 20), pady=15)
        
        # Video Display Area
        video_container = tk.Frame(
            main_container,
            bg=self.theme["bg_tertiary"],
            relief="flat",
            bd=2
        )
        video_container.pack(fill="both", expand=True, pady=(0, 20))
        
        video_header = tk.Frame(video_container, bg=self.theme["bg_tertiary"])
        video_header.pack(fill="x", padx=20, pady=(15, 10))
        
        tk.Label(
            video_header,
            text="LIVE VIDEO FEED",
            font=self.header_font,
            fg=self.theme["text_primary"],
            bg=self.theme["bg_tertiary"]
        ).pack(side="left")
        
        self.fps_label = tk.Label(
            video_header,
            text="FPS: --",
            font=self.body_font,
            fg=self.theme["text_secondary"],
            bg=self.theme["bg_tertiary"]
        )
        self.fps_label.pack(side="right")
        
        # Video display
        self.video_label = tk.Label(
            video_container,
            bg="black",
            relief="sunken",
            bd=1
        )
        self.video_label.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Status Bar
        status_frame = tk.Frame(
            main_container,
            bg=self.theme["bg_secondary"],
            height=60
        )
        status_frame.pack(fill="x")
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="System Ready • Click 'START DETECTION' to begin")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=self.body_font,
            fg=self.theme["text_primary"],
            bg=self.theme["bg_secondary"],
            anchor="w",
            padx=20
        )
        status_label.pack(fill="both", expand=True)
        
        # Configure ttk styles
        self.configure_styles()

    def configure_styles(self):
        """Configure ttk widget styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure(
            "TButton",
            padding=10,
            relief="flat",
            borderwidth=0,
            font=self.body_font
        )
        
        style.configure(
            "Accent.TButton",
            background=self.theme["accent"],
            foreground="white",
            borderwidth=0,
            focuscolor="none"
        )
        
        style.map(
            "Accent.TButton",
            background=[("active", self.theme["warning"]), ("disabled", "#666666")],
            foreground=[("active", "white"), ("disabled", "#999999")]
        )
        
        style.configure(
            "Admin.TButton",
            background=self.theme["info"],
            foreground="white",
            borderwidth=0
        )
        
        style.map(
            "Admin.TButton",
            background=[("active", "#1976D2"), ("disabled", "#666666")]
        )

    # ================= MODEL MANAGEMENT =================
    def load_models(self):
        """Load AI models (lazy loading)"""
        if not self.model_loaded:
            try:
                self.status_var.set("Loading detection model...")
                self.model = YOLO("best.pt")
                self.model.overrides["verbose"] = False
                
                self.status_var.set("Loading OCR engine...")
                self.reader = easyocr.Reader(["en"], gpu=False)
                
                self.model_loaded = True
                return True
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load models: {str(e)}")
                return False
        return True

    # ================= ADMIN PANEL =================
    def admin_login(self):
        """Admin login dialog"""
        login_window = tk.Toplevel(self.root)
        login_window.title("Administrator Login")
        login_window.geometry("400x300")
        login_window.configure(bg=self.theme["bg_primary"])
        login_window.resizable(False, False)
        login_window.transient(self.root)
        login_window.grab_set()
        
        # Center login window
        login_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (400 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (300 // 2)
        login_window.geometry(f"400x300+{x}+{y}")
        
        # Login UI
        login_frame = tk.Frame(login_window, bg=self.theme["bg_primary"])
        login_frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        tk.Label(
            login_frame,
            text="ADMINISTRATOR LOGIN",
            font=self.header_font,
            fg=self.theme["accent"],
            bg=self.theme["bg_primary"]
        ).pack(pady=(0, 30))
        
        # Username
        tk.Label(
            login_frame,
            text="Username",
            font=self.body_font,
            fg=self.theme["text_secondary"],
            bg=self.theme["bg_primary"]
        ).pack(anchor="w", pady=(0, 5))
        
        username_entry = ttk.Entry(login_frame, font=self.body_font)
        username_entry.pack(fill="x", pady=(0, 15))
        username_entry.focus()
        
        # Password
        tk.Label(
            login_frame,
            text="Password",
            font=self.body_font,
            fg=self.theme["text_secondary"],
            bg=self.theme["bg_primary"]
        ).pack(anchor="w", pady=(0, 5))
        
        password_entry = ttk.Entry(login_frame, font=self.body_font, show="•")
        password_entry.pack(fill="x", pady=(0, 30))
        
        def attempt_login():
            username = username_entry.get()
            password = password_entry.get()
            
            if (username == CONFIG["ADMIN_CREDENTIALS"]["username"] and 
                password == CONFIG["ADMIN_CREDENTIALS"]["password"]):
                login_window.destroy()
                self.open_admin_panel()
            else:
                messagebox.showerror(
                    "Authentication Failed",
                    "Invalid username or password",
                    parent=login_window
                )
                password_entry.delete(0, tk.END)
        
        # Bind Enter key to login
        password_entry.bind('<Return>', lambda e: attempt_login())
        
        # Login button
        login_btn = ttk.Button(
            login_frame,
            text="LOGIN",
            command=attempt_login,
            style="Accent.TButton"
        )
        login_btn.pack(fill="x")

    def open_admin_panel(self):
        """Open admin management panel"""
        admin_window = tk.Toplevel(self.root)
        admin_window.title("Vehicle Database Management")
        admin_window.geometry("600x500")
        admin_window.configure(bg=self.theme["bg_primary"])
        admin_window.minsize(500, 400)
        
        # Center window
        admin_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (600 // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (500 // 2)
        admin_window.geometry(f"600x500+{x}+{y}")
        
        # Admin UI
        notebook = ttk.Notebook(admin_window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add Vehicle Tab
        add_frame = ttk.Frame(notebook)
        notebook.add(add_frame, text="➕ Add Vehicle")
        
        tk.Label(
            add_frame,
            text="Register New Vehicle",
            font=self.header_font,
            fg=self.theme["accent"]
        ).pack(pady=(20, 30))
        
        # Form fields
        fields = [
            ("License Plate Number", "plate"),
            ("Origin (From)", "from"),
            ("Destination (To)", "to")
        ]
        
        entries = {}
        for label_text, key in fields:
            tk.Label(
                add_frame,
                text=label_text,
                font=self.body_font
            ).pack(pady=(0, 5))
            
            entry = ttk.Entry(add_frame, font=self.body_font, width=40)
            entry.pack(pady=(0, 15))
            entries[key] = entry
        
        def save_vehicle():
            plate = normalize_plate(entries["plate"].get())
            from_place = entries["from"].get().strip()
            to_place = entries["to"].get().strip()
            
            if not plate or len(plate) < 4:
                messagebox.showerror("Invalid Input", "Please enter a valid license plate number")
                return
            
            if not from_place or not to_place:
                messagebox.showerror("Invalid Input", "Please enter both origin and destination")
                return
            
            if self.vehicle_db.add_vehicle(plate, from_place, to_place):
                messagebox.showinfo("Success", f"Vehicle {plate} registered successfully!")
                for entry in entries.values():
                    entry.delete(0, tk.END)
                entries["plate"].focus()
            else:
                messagebox.showerror("Error", "Failed to save vehicle data")
        
        ttk.Button(
            add_frame,
            text="SAVE VEHICLE",
            command=save_vehicle,
            style="Accent.TButton"
        ).pack(pady=20)
        
        # View Database Tab
        view_frame = ttk.Frame(notebook)
        notebook.add(view_frame, text="👁️ View Database")
        
        # Database list with scrollbar
        list_frame = tk.Frame(view_frame)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.db_listbox = tk.Listbox(
            list_frame,
            font=self.mono_font,
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            selectbackground=self.theme["accent"],
            yscrollcommand=scrollbar.set,
            relief="flat",
            bd=0
        )
        self.db_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.db_listbox.yview)
        
        self.refresh_database_list()
        
        # Refresh button
        ttk.Button(
            view_frame,
            text="🔄 Refresh List",
            command=self.refresh_database_list
        ).pack(pady=10)

    def refresh_database_list(self):
        """Refresh the database list display"""
        self.db_listbox.delete(0, tk.END)
        for plate, info in self.vehicle_db.db.items():
            entry = f"{plate:15} | From: {info['from']:20} | To: {info['to']:20}"
            self.db_listbox.insert(tk.END, entry)

    # ================= CAMERA CONTROL =================
    def start_camera(self):
        """Start the camera and detection"""
        if self.running:
            return
        
        if not self.load_models():
            return
        
        self.running = True
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera. Please check your camera connection.")
            self.running = False
            return
        
        # Update UI state
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("System Active • Detecting license plates...")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

    def stop_camera(self):
        """Stop the camera and detection"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Update UI state
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.video_label.config(image="")
        self.status_var.set("System Ready • Detection stopped")

    def detection_loop(self):
        """Main detection loop"""
        fps_counter = 0
        fps_timer = time.time()
        
        while self.running and self.cap and self.cap.isOpened():
            start_time = time.time()
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Resize for better performance
            frame = cv2.resize(frame, (800, 600))
            display_frame = frame.copy()
            
            # Run detection
            try:
                results = self.model.predict(frame, conf=CONFIG["MODEL_CONFIDENCE"], verbose=False)
                
                for box in results[0].boxes or []:
                    bbox = box.xyxy[0].cpu().numpy()
                    raw_text = get_ocr_text(display_frame, bbox, self.reader)
                    
                    if not raw_text:
                        continue
                    
                    plate = normalize_plate(raw_text)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    if plate in self.vehicle_db.db:
                        # Known vehicle
                        vehicle_info = self.vehicle_db.db[plate]
                        label = f"{plate} ✅ AUTHORIZED"
                        color = (76, 175, 80)  # Green
                        self.status_var.set(f"✅ Authorized: {plate} | {vehicle_info['from']} → {vehicle_info['to']}")
                        
                        # Open map if not already opened for this plate
                        if self.last_map_opened != plate:
                            url = f"https://www.google.com/maps/dir/{vehicle_info['from']}/{vehicle_info['to']}"
                            webbrowser.open(url)
                            self.last_map_opened = plate
                    else:
                        # Unknown vehicle
                        label = f"{plate} ⚠️ UNKNOWN"
                        color = (244, 67, 54)  # Red
                        self.status_var.set(f"⚠️ Unknown Vehicle Detected: {plate}")
                    
                    # Draw bounding box and label
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(display_frame, (x1, y1-35), (x1+len(label)*12, y1), color, -1)
                    cv2.putText(display_frame, label, (x1+5, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            except Exception as e:
                print(f"Detection error: {e}")
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.root.after(0, self.update_fps, fps_counter)
                fps_counter = 0
                fps_timer = time.time()
            
            # Convert to RGB for Tkinter
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to ImageTk
            pil_image = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            
            # Update display in main thread
            self.root.after(0, self.update_display, imgtk)
            
            # Control frame rate
            elapsed = time.time() - start_time
            time.sleep(max(0.03 - elapsed, 0))

    def update_display(self, image):
        """Update the video display (must be called from main thread)"""
        if self.running:
            self.video_label.imgtk = image
            self.video_label.config(image=image)

    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.config(text=f"FPS: {fps}")

    def on_closing(self):
        """Handle application closing"""
        self.stop_camera()
        if self.vehicle_db:
            self.vehicle_db.save_database()
        self.root.destroy()

# ================= APPLICATION ENTRY POINT =================
if __name__ == "__main__":
    root = tk.Tk()
    
    # Set application icon (if available)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = NumberPlateApp(root)
    
    # Set application to always on top initially (optional)
    # root.attributes('-topmost', 1)
    # root.after(100, lambda: root.attributes('-topmost', 0))
    
    root.mainloop()