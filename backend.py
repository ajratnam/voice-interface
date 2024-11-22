import requests
import socket
import json
import struct
import whisper
import tempfile
import os
import threading
import socketio
from pyngrok import ngrok
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class WhisperServer:
    def __init__(self, socketio_server, host='localhost', port=65432):
        self.host = host
        self.port = port
        self.socketio_server = socketio_server
        self.model = None
        self.model_name = "tiny"
        self.sio = socketio.Client()
        self.setup_socketio()
        self.chunk_size = 8192
        self.session_id = None
        self.file_hash_map = {}
        self.load_model(self.model_name)

    def setup_socketio(self):
        try:
            self.sio.connect(self.socketio_server)
            print(f"Connected to Socket.IO server at {self.socketio_server}")
        except Exception as e:
            print(f"Error connecting to Socket.IO server: {e}")

    def send_chunked(self, conn, data):
        """Send data to the client in chunks."""
        try:
            send_json_data = json.dumps(data).encode()
            size = len(send_json_data)
            conn.sendall(struct.pack('!I', size))
            conn.sendall(send_json_data)
        except Exception as e:
            print(f"Error in send_chunked: {e}")
            raise

    def receive_chunked(self, conn):
        """Receive data from the client in chunks."""
        try:
            size_data = conn.recv(4)
            if not size_data:
                return None
            total_size = struct.unpack('!I', size_data)[0]
            data = conn.recv(total_size)
            if not data:
                return None
            return json.loads(data)
        except Exception as e:
            print(f"Error in receive_chunked: {e}")
            return None

    def load_model(self, model_name):
        try:
            if not self.model or self.model_name != model_name:
                self.model = whisper.load_model(model_name)
                self.model_name = model_name
            return {"status": "success", "message": f"Loaded {model_name} model successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def download_audio(self, audio_url):
        """Download audio file from the provided URL."""
        try:
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                return temp_file.name
        except Exception as e:
            print(f"Error downloading audio file: {e}")
            return None

    def send_output(self, mode, content):
        mode = {"document": "pentest-vui", "chat": "chat", "create": "pentest-vui"}[mode]
        try:
            self.sio.emit(mode, {"session_id": "SESSION", "message": content})
            print(f"Sent event '{mode}' with content to Socket.IO server")
        except Exception as e:
            print(f"Error sending event to Socket.IO server: {e}")

    def detect_language(self, audio_url):
        """Detect the language of the uploaded audio file."""
        temp_path = self.download_audio(audio_url)
        if not temp_path:
            return {"status": "error", "message": "Failed to download audio file"}

        try:
            audio = whisper.load_audio(temp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            return {"status": "success", "language": detected_lang}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            os.unlink(temp_path)

    def transcribe_audio(self, audio_url, mode):
        """Transcribe the uploaded audio file."""
        temp_path = self.download_audio(audio_url)
        if not temp_path:
            return {"status": "error", "message": "Failed to download audio file"}

        try:
            audio = whisper.load_audio(temp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            options = whisper.DecodingOptions()
            result = whisper.decode(self.model, mel, options)
            self.send_output(mode, result.text)
            return {"status": "success", "text": result.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            os.unlink(temp_path)

    def translate_audio(self, audio_url, mode):
        """Translate the uploaded audio file."""
        temp_path = self.download_audio(audio_url)
        if not temp_path:
            return {"status": "error", "message": "Failed to download audio file"}

        try:
            audio = whisper.load_audio(temp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            options = whisper.DecodingOptions(task="translate")
            result = whisper.decode(self.model, mel, options)
            return {"status": "success", "text": result.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            os.unlink(temp_path)

    def handle_client(self, conn, addr):
        """Handle incoming client connections."""
        print(f"New connection from {addr}")
        while True:
            try:
                message = self.receive_chunked(conn)
                if not message:
                    break

                command = message.get("command")
                audio_url = message.get("audio_url")
                mode = message.get("mode", "document")

                if command == "load_model":
                    response = self.load_model(message["model_name"])
                elif command == "detect_language":
                    response = self.detect_language(audio_url)
                elif command == "transcribe":
                    response = self.transcribe_audio(audio_url, mode)
                elif command == "translate":
                    response = self.translate_audio(audio_url, mode)
                else:
                    response = {"status": "error", "message": "Unknown command"}

                self.send_chunked(conn, response)

            except Exception as e:
                print(f"Error handling client: {e}")
                break

        conn.close()
        print(f"Connection closed from {addr}")

    def start_tcp_server(self):
        """Start the TCP server."""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen()
        print(f"TCP server listening on {self.host}:{self.port}")
        while True:
            conn, addr = self.server.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()

    def start_localtunnel(self):
        print("Starting localtunnel...")
        public_url = ngrok.connect(65432, "tcp").public_url
        url_parts = public_url.split("://")[-1].split(":")
        hostname = url_parts[0]
        port = int(url_parts[1])

        # MongoDB connection
        try:
            mongo_client = MongoClient(os.getenv("MONGO_URI"))
            db = mongo_client["PROJECT"]
            ngrok_collection = db["ngrok"]

            # Clear the collection and insert the new document
            ngrok_collection.delete_many({})
            ngrok_collection.insert_one({
                "url": hostname,
                "port": port
            })
            print(f"Updated MongoDB with Ngrok URL: {hostname} and port: {port}")
        except Exception as e:
            print(f"Error updating MongoDB: {e}")

    def start(self):
        # Start localtunnel in the main thread
        self.start_localtunnel()
        self.start_tcp_server()


if __name__ == "__main__":
    server = WhisperServer(socketio_server="http://someotherwebsite.com")
    server.start()
