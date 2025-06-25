import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageGrab
import io
import base64
import json

def process_image(img):
    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Keine Linie gefunden!")
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)

    linie = np.full(width, np.nan)
    for x in range(width):
        col = mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            linie[x] = np.mean(ys)
    
    nans, = np.isnan(linie).nonzero()
    not_nans = np.isfinite(linie)
    linie[nans] = np.interp(nans, np.where(not_nans)[0], linie[not_nans])

    linie_min = np.min(linie)
    linie_max = np.max(linie)
    linie_norm = 1 - (linie - linie_min) / (linie_max - linie_min)

    valid_x = np.where(~np.isnan(linie))[0]
    start_x = valid_x[0]
    end_x = valid_x[-1]
    linienbreite = end_x - start_x

    left_special = int(start_x + linienbreite * (17/365))
    right_special = int(end_x - linienbreite * (13/365))
    middle_edges = np.linspace(left_special, right_special, 12, dtype=int)[1:-1]
    compartment_edges = [start_x, left_special] + list(middle_edges) + [right_special, end_x]

    img_disp = img.copy()
    for x in range(width):
        y = int(linie[x])
        cv2.circle(img_disp, (x, y), 1, (255, 0, 255), -1)
    for x in compartment_edges:
        cv2.line(img_disp, (x, 0), (x, height-1), (0, 0, 255), 1)

    schnittpunkte = []
    for x in compartment_edges:
        y_norm = 1 - (linie[x] - linie_min) / (linie_max - linie_min)
        schnittpunkte.append((int(x), float(y_norm)))

    img_disp_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_disp_rgb)
    return pil_img, schnittpunkte

class ImageScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Datenextraktion")
        self.root.geometry("500x500")
        
        main_frame = tk.Frame(root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        instruction = tk.Label(main_frame, text="Hier Bild mit Strg+V einfügen'")
        instruction.pack(pady=10)
        
        self.image_label = tk.Label(main_frame)
        self.image_label.pack(pady=10)
        
        process_button = tk.Button(main_frame, text="Verarbeiten", command=self.process_image)
        process_button.pack(pady=10)
        
        self.current_image = None
        self.schnittpunkte_label = tk.Label(main_frame, text="")
        self.schnittpunkte_label.pack(pady=10)
        
        self.root.bind('<Control-v>', self.paste_image)
        
    def paste_image(self, event):
        try:
            image = ImageGrab.grabclipboard()
            if image is not None:
                display_size = (400, 300)
                image.thumbnail(display_size, Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                self.current_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                return
            image = self.root.clipboard_get()
            if image.startswith('data:image'):
                image_data = image.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = Image.open(image)
            display_size = (400, 300)
            image.thumbnail(display_size, Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.current_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Einfügen des Bildes: {str(e)}")
    
    def process_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warnung", "Bitte fügen Sie zuerst ein Bild ein!")
            return
        
        try:
            pil_img, schnittpunkte = process_image(self.current_image)
            display_size = (400, 300)
            pil_img.thumbnail(display_size, Image.ANTIALIAS)
            self.tk_image = ImageTk.PhotoImage(pil_img)
            self.image_label.configure(image=self.tk_image)
            self.image_label.image = self.tk_image

            img = self.current_image
            height, width = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            linie = np.full(width, np.nan)
            for x in range(width):
                col = mask[:, x]
                ys = np.where(col > 0)[0]
                if len(ys) > 0:
                    linie[x] = np.mean(ys)
            nans, = np.isnan(linie).nonzero()
            not_nans = np.isfinite(linie)
            linie[nans] = np.interp(nans, np.where(not_nans)[0], linie[not_nans])
            linie_min = np.min(linie)
            linie_max = np.max(linie)
            linie_norm = 1 - (linie - linie_min) / (linie_max - linie_min)
            valid_x = np.where(~np.isnan(linie))[0]
            start_x = valid_x[0]
            end_x = valid_x[-1]
            linienbreite = end_x - start_x
            left_special = int(start_x + linienbreite * (17/365))
            right_special = int(end_x - linienbreite * (13/365))
            middle_edges = np.linspace(left_special, right_special, 12, dtype=int)[1:-1]
            compartment_edges = [start_x, left_special] + list(middle_edges) + [right_special, end_x]
            
            monate = [
                "Juni 2025", "Mai 2025", "April 2025", "März 2025", "Februar 2025", "Januar 2025",
                "Dezember 2024", "November 2024", "Oktober 2024", "September 2024", "August 2024", "Juli 2024",
                "Juni 2024", "Mai 2024"
            ]
            
            followermonate = [
                "Juni 2024", "Juli 2024", "August 2024", "September 2024", "Oktober 2024", "November 2024",
                "Dezember 2024", "Januar 2025", "Februar 2025", "März 2025", "April 2025", "Mai 2025",
                "Juni 2025"
            ]
            
            n_compartments = len(compartment_edges) - 1
            followermonate = followermonate[:n_compartments]
            schnittpunkte_text = ""
            mai_index = followermonate.index("Mai 2025") if "Mai 2025" in followermonate else None
            mai_diff = None
            if mai_index is not None:
                x_links = compartment_edges[mai_index]
                x_rechts = compartment_edges[mai_index + 1]
                y_links = linie_norm[x_links]
                y_rechts = linie_norm[x_rechts]
                mai_diff = y_rechts - y_links
                schnittpunkte_text = f"Mai 2025: y_links={y_links:.3f}, y_rechts={y_rechts:.3f}, Unterschied={mai_diff:.3f}"
            self.schnittpunkte_label.config(text=schnittpunkte_text)
            
            def submit_callback():
                text = textbox.get("1.0", tk.END)
                lines_raw = text.splitlines()
                format_error_msg = "Das Format stimmt nicht! Bitte kopiere die Daten exakt wie aus der Tabelle ein."
                if lines_raw[0].strip() != "":
                    messagebox.showerror("Fehler", format_error_msg)
                    return
                if len(lines_raw) != 22:
                    messagebox.showerror("Fehler", format_error_msg)
                    return
                if lines_raw[-1].strip() != "2024":
                    messagebox.showerror("Fehler", format_error_msg)
                    return
                lines = [l.strip() for l in lines_raw if l.strip().endswith('%') and not l.strip().replace('%','').replace(',','.').replace('-','').isdigit()]
                values = [float(l.replace('%','').replace(',','.')) for l in lines]
                if len(values) > 6:
                    values = values[:6] + values[7:]
                out_monate = monate[2:14]
                out_values = values[2:14]
                if out_values:
                    avg = sum(out_values) / len(out_values)
                    avg = avg * 12  # Multipliziere den Durchschnitt mit 12
                    april_value = None
                    if "April 2025" in out_monate:
                        idx_april = out_monate.index("April 2025")
                        april_value = out_values[idx_april]
                    try:
                        with open('data.json', 'r') as f:
                            data = json.load(f)
                        
                        new_entry = {
                            "avg": round(avg, 2),
                            "april": round(april_value, 2) if april_value is not None else None,
                            "follower": round(mai_diff, 3) if mai_diff is not None else None
                        }
                        
                        data["entries"].append(new_entry)
                        
                        with open('data.json', 'w') as f:
                            json.dump(data, f, indent=2)
                            
                        print(f"Aktuelle Anzahl der Einträge: {len(data['entries'])}")
                            
                        plt.close('all')
                        fenster.destroy()
                        
                        self.image_label.configure(image='')
                        self.image_label.image = None
                        self.schnittpunkte_label.config(text="")
                        self.current_image = None
                        
                    except Exception as e:
                        messagebox.showerror("Fehler", f"Fehler beim Speichern in data.json: {str(e)}")
            
            fenster = tk.Toplevel(self.root)
            fenster.title("Datenextraktion")
            try:
                fenster.iconbitmap('')
            except Exception:
                pass
            fenster.geometry("400x400+500+500")  # Textfeld unter dem Hauptfenster
            label = tk.Label(fenster, text="Performancedaten hier einfügen:")
            label.pack(pady=10)
            textbox = tk.Text(fenster, height=15, width=30)
            textbox.pack(pady=10)
            
            button_frame = tk.Frame(fenster)
            button_frame.pack(pady=10)
            
            submit_btn = tk.Button(button_frame, text="Submit", command=submit_callback)
            submit_btn.pack(side=tk.LEFT, padx=5)
            
            def reset_callback():
                plt.close('all')
                fenster.destroy()
                
                self.image_label.configure(image='')
                self.image_label.image = None
                self.schnittpunkte_label.config(text="")
                self.current_image = None
            
            reset_btn = tk.Button(button_frame, text="Reset", command=reset_callback)
            reset_btn.pack(side=tk.LEFT, padx=5)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.arange(width), linie_norm, label='Normierte Linie', color='blue')
            for x in compartment_edges:
                ax.axvline(x, color='red', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_xlabel('x (Pixel)')
            ax.set_ylabel('Normierte Höhe')
            ax.set_title('Normierte Linie mit Compartments')
            ax.legend()
            plt.tight_layout()
            try:
                mng = plt.get_current_fig_manager()
                mng.window.wm_geometry("+500+0")  # Graph oben in der Mitte
            except Exception:
                pass
            plt.show()
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei der Verarbeitung: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageScannerApp(root)
    root.mainloop()
