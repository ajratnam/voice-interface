import re
import tkinter as tk
from io import BytesIO
from tkinter import filedialog, messagebox, ttk

import requests
import sounddevice as sd
import numpy as np
import wavio
import os
import socket
import json
import base64
import struct
from threading import Thread, Event
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

json_reg = re.compile(r"{.*}")


class WhisperClient:
    def __init__(self):
        self.root = tk.Tk()
        self.chunk_size = 8192
        self.setup_gui()
        self.recording_event = Event()
        self.audio_path = None
        self.setup_socket()
        self.sent = False
        self.mode = "document"

    def send_chunked(self, data):
        try:
            send_json_data = json.dumps(data).encode()
            print(f"Encoded data size with padding: {len(send_json_data)}")

            size = len(send_json_data)
            self.sock.sendall(struct.pack('!I', size))
            self.sock.sendall(send_json_data)  # Send all at once for efficiency

        except Exception as e:
            print(f"Error in send_chunked: {e}")
            raise

    def receive_chunked(self):
        try:
            size_data = self.sock.recv(4)
            if not size_data:
                return None
            total_size = struct.unpack('!I', size_data)[0]
            data = self.sock.recv(total_size)  # Receive all at once
            if not data:
                return None

            received_data = json.loads(data.decode())
            return received_data

        except Exception as e:
            print(f"Error during data transfer: {e}")
            return None

    def get_ngrok_details(self):
        try:
            # Connect to MongoDB and retrieve the Ngrok details
            mongo_client = MongoClient(os.getenv("MONGO_URI"))
            db = mongo_client["PROJECT"]
            ngrok_collection = db["ngrok"]

            ngrok_details = ngrok_collection.find_one()
            print(f"Ngrok details: {ngrok_details}")
            if ngrok_details:
                return ngrok_details["url"], ngrok_details["port"]
            else:
                raise Exception("No Ngrok details found in MongoDB.")
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching Ngrok details: {e}")
            return None, None

    def setup_socket(self):
        # Get the Ngrok URL and port dynamically from MongoDB
        hostname, port = self.get_ngrok_details()
        if not hostname or not port:
            self.root.quit()
            return

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((hostname, port))
        except ConnectionRefusedError:
            messagebox.showerror("Error", "Could not connect to server. Please ensure the server is running.")
            self.root.quit()

    def send_command(self, command, **kwargs):
        mode = self.mode_var.get()
        message = {"command": command, "mode": mode, **kwargs}
        # print(f"Sending command: {message}")
        try:
            self.send_chunked(message)
            response = self.receive_chunked()
            print(f"Received response: {response}")
            if mode in ["transcribe", "translate"]:
                self.sent = True
            return response
        except Exception as e:
            print(f"Error in send_command: {e}")
            return {"status": "error", "message": str(e)}

    def setup_gui(self):
        # GUI setup code remains the same
        self.root.title("LexVon")
        self.root.geometry("400x650")
        self.root.configure(bg="#1c1c1e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Arial", 10), padding=6, relief="flat", background="#4CAF50",
                        foreground="#FFFFFF")
        style.map("TButton", background=[("active", "#357a38")])
        style.configure("TLabel", font=("Arial", 12), background="#1c1c1e", foreground="#f2f2f2")
        style.configure("TFrame", background="#333333")
        style.configure("green.Horizontal.TProgressbar", troughcolor="#333333", background="#4CAF50")

        mode_frame = ttk.Frame(self.root)
        mode_frame.pack(pady=(20, 10))

        mode_label = ttk.Label(mode_frame, text="Mode:")
        mode_label.pack(side="left", padx=5)

        self.mode_var = tk.StringVar(value="document")
        self.mode_switch = ttk.Checkbutton(
            mode_frame,
            text="Document/Chat",
            variable=self.mode_var,
            onvalue="chat",
            offvalue="document",
            command=self.update_mode
        )
        self.mode_switch.pack(side="left", padx=5)

        self.model_options = ["", "tiny", "base", "small", "medium", "large", "turbo"]
        self.selected_model = self.model_options[-2]

        model_label = ttk.Label(self.root, text="Select Model:")
        model_label.pack(pady=(10, 5))

        self.model_var = tk.StringVar(self.root)
        self.model_var.set(self.selected_model)
        model_menu = ttk.OptionMenu(self.root, self.model_var, *self.model_options, command=self.update_model)
        model_menu.pack(pady=(0, 20))
        button_frame1 = ttk.Frame(self.root)
        button_frame1.pack(pady=(0, 20))

        self.record_button = ttk.Button(button_frame1, text="🎙️ Record Audio", command=self.toggle_recording)
        self.record_button.pack(side="left", padx=5)

        open_button = ttk.Button(button_frame1, text="📂 Open Audio File", command=self.open_audio_file)
        open_button.pack(side="left", padx=5)

        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", length=250,
                                            style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=10)

        button_frame2 = ttk.Frame(self.root)
        button_frame2.pack(pady=(10, 5))

        transcribe_button = ttk.Button(button_frame2, text="Transcribe", command=self.transcribe_audio)
        transcribe_button.pack(side="left", padx=5)

        translate_button = ttk.Button(button_frame2, text="Translate", command=self.translate_text)
        translate_button.pack(side="left", padx=5)

        self.status_label = ttk.Label(self.root, text="")
        self.status_label.pack(pady=(10, 10))

        self.language_label = ttk.Label(self.root, text="Detected Language: None")
        self.language_label.pack(pady=5)

        self.transcript_frame = ttk.Frame(self.root)
        transcript_label = ttk.Label(self.transcript_frame, text="Transcription:")
        transcript_label.pack(anchor="w")
        self.transcript_text = tk.Text(self.transcript_frame, wrap="word", font=("Arial", 10), height=10, width=50)
        self.transcript_text.pack(padx=5, pady=5)

        self.translation_frame = ttk.Frame(self.root)
        translation_label = ttk.Label(self.translation_frame, text="Translation:")
        translation_label.pack(anchor="w")
        self.translation_text = tk.Text(self.translation_frame, wrap="word", font=("Arial", 10), height=10, width=50)
        self.translation_text.pack(padx=5, pady=5)

        self.clear_button = ttk.Button(self.root, text="Clear Text Fields", command=self.clear_text_fields)

        # self.update_model(self.selected_model)

    def update_model(self, selected):
        Thread(target=self._update_model_thread, args=(selected,)).start()

    def _update_model_thread(self, selected):
        self.progress_bar.start()
        try:
            response = self.send_command("load_model", model_name=selected)
            if response["status"] == "success":
                self.status_label.config(text=response["message"])
            else:
                self.status_label.config(text=f"Error: {response['message']}")
        finally:
            self.progress_bar.stop()

    def upload_file(self, base64_data, filename="audio.mp3"):
        """Uploads a file to tmpfiles.org and returns the direct download URL."""
        binary_data = base64.b64decode(base64_data)
        file_data = BytesIO(binary_data)
        files = {'file': (filename, file_data, 'audio/mpeg')}

        try:
            response = requests.post("https://tmpfiles.org/api/v1/upload", files=files)
            response.raise_for_status()
            data = response.json()
            temp_url = data['data']['url']
            url_parts = temp_url.split("/", 3)
            direct_url = f"{url_parts[0]}//{url_parts[2]}/dl/{url_parts[3]}"
            return direct_url
        except requests.exceptions.RequestException as e:
            print(f"Error uploading file: {e}")
            return None

    def detect_language(self, audio_file):
        """Detects language from the audio file by uploading it and sending the link."""
        try:
            with open(audio_file, 'rb') as f:
                base64_audio = base64.b64encode(f.read()).decode()

            upload_url = self.upload_file(base64_audio, os.path.basename(audio_file))
            if not upload_url:
                self.status_label.config(text="Error uploading audio file for language detection.")
                return

            response = self.send_command("detect_language", audio_url=upload_url)
            if response["status"] == "success":
                self.language_label.config(text=f"Detected Language: {response['language']}")
                self.status_label.config(text=f"Loaded file: {os.path.basename(audio_file)}")
            else:
                self.status_label.config(text=f"Error: {response['message']}")
        except Exception as e:
            self.status_label.config(text=f"Error during language detection: {e}")

    def start_recording(self):
        fs = 44100

        def record():
            try:
                self.status_label.config(text="Recording... Press Stop to finish.")
                self.recording_event.set()
                buffer = []

                def callback(indata, frames, time, status):
                    if self.recording_event.is_set():
                        buffer.append(indata.copy())
                    else:
                        raise sd.CallbackStop

                with sd.InputStream(samplerate=fs, channels=1, callback=callback):
                    while self.recording_event.is_set():
                        sd.sleep(100)

                recording = np.concatenate(buffer, axis=0)
                save_path = os.path.join("recordings", "recording.wav")
                os.makedirs("recordings", exist_ok=True)
                wavio.write(save_path, recording, fs, sampwidth=2)
                self.audio_path = save_path
                self.status_label.config(text="Audio recorded successfully!")
                self.detect_language(self.audio_path)
            except Exception as e:
                self.status_label.config(text=f"Error during recording: {e}")

        self.recording_thread = Thread(target=record, daemon=True)
        self.recording_thread.start()

    def toggle_recording(self):
        if self.recording_event.is_set():
            self.stop_recording()
        else:
            self.start_recording()
            self.record_button.config(text="⏹️ Stop Recording")

    def stop_recording(self):
        self.recording_event.clear()
        self.record_button.config(text="🎙️ Record Audio")
        self.status_label.config(text="Recording stopped. Processing audio...")

    def open_audio_file(self):
        file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
        if file:
            self.audio_path = file
            self.detect_language(self.audio_path)

    def transcribe_audio(self):
        Thread(target=self._transcribe_audio_thread).start()

    def update_mode(self):
        self.mode = self.mode_var.get()
        self.status_label.config(text=f"Mode switched to: {self.mode}")

    def _transcribe_audio_thread(self):
        self.progress_bar.start()
        try:
            if self.audio_path is None:
                messagebox.showwarning("Warning", "Please record or open an audio file first.")
                return

            with open(self.audio_path, 'rb') as f:
                base64_audio = base64.b64encode(f.read()).decode()

            upload_url = self.upload_file(base64_audio, os.path.basename(self.audio_path))
            if not upload_url:
                self.status_label.config(text="Error uploading audio file for transcription.")
                return

            response = self.send_command("transcribe", audio_url=upload_url)
            if response["status"] == "success":
                self.transcript_text.delete("1.0", tk.END)
                self.transcript_text.insert(tk.END, response["text"])
                self.translation_frame.pack_forget()
                self.transcript_frame.pack(pady=(5, 10))
                self.status_label.config(text=f"Transcription completed in {self.mode} mode.")
                self.clear_button.pack(pady=(10, 20))
            else:
                self.status_label.config(text=f"Error: {response['message']}")
        finally:
            self.progress_bar.stop()

    def translate_text(self):
        Thread(target=self._translate_text_thread).start()

    def _translate_text_thread(self):
        self.progress_bar.start()
        try:
            if self.audio_path is None:
                messagebox.showwarning("Warning", "Please record or open an audio file first.")
                return

            with open(self.audio_path, 'rb') as f:
                base64_audio = base64.b64encode(f.read()).decode()

            upload_url = self.upload_file(base64_audio, os.path.basename(self.audio_path))
            if not upload_url:
                self.status_label.config(text="Error uploading audio file for translation.")
                return

            response = self.send_command("translate", audio_url=upload_url)
            if response["status"] == "success":
                self.translation_text.delete("1.0", tk.END)
                self.translation_text.insert(tk.END, response["text"])
                self.transcript_frame.pack_forget()
                self.translation_frame.pack(pady=(5, 10))
                self.status_label.config(text=f"Translation completed in {self.mode} mode.")
                self.clear_button.pack(pady=(10, 20))
            else:
                self.status_label.config(text=f"Error: {response['message']}")
        finally:
            self.progress_bar.stop()

    def clear_text_fields(self):
        self.sent = False
        self.transcript_text.delete("1.0", tk.END)
        self.translation_text.delete("1.0", tk.END)
        self.transcript_frame.pack_forget()
        self.translation_frame.pack_forget()
        self.clear_button.pack_forget()
        self.status_label.config(text="Text fields cleared.")

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.sock.close()


if __name__ == "__main__":
    client = WhisperClient()
    client.run()
