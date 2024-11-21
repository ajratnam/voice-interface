import re
import socket
import json
import uuid
from hashlib import sha512
import struct
import whisper
import tempfile
import os
import base64
import threading
import socketio
import heapq
from collections import Counter
from pyngrok import ngrok
from pymongo import MongoClient


json_reg = re.compile(r"{.*}")


class HuffmanNode:
    def __init__(self, left=None, right=None, char=None, weight=0):
        self.left = left
        self.right = right
        self.char = char
        self.weight = weight

    def walk(self, code, acc):
        if self.char is not None:
            code[self.char] = acc or "0"
        else:
            self.left.walk(code, acc + "0")
            self.right.walk(code, acc + "1")

    def __lt__(self, other):  # IMPORTANT FIX: Add comparison logic
        return self.weight < other.weight


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
        self.huffman_cache = {}
        self.load_model(self.model_name)

    def huffman_encode(self, data):
        if data in self.huffman_cache:
            return self.huffman_cache[data]

        freq = Counter(data)
        heap = [(freq[char], HuffmanNode(char=char, weight=freq[char])) for char in freq]
        heapq.heapify(heap)

        while len(heap) > 1:
            w1, n1 = heapq.heappop(heap)
            w2, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (w1 + w2, HuffmanNode(left=n1, right=n2, weight=w1 + w2)))

        code = {}
        [(_, root)] = heap
        root.walk(code, "")

        encoded_data = "".join(code[char] for char in data)
        self.huffman_cache[data] = encoded_data, code
        return encoded_data, code

    def huffman_decode(self, encoded_data, code):
        reverse_code = {value: key for key, value in code.items()}
        current_code = ""
        decoded_data = []
        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_code:
                decoded_data.append(reverse_code[current_code])
                current_code = ""
        return bytes(map(int, decoded_data))

    def setup_socketio(self):
        try:
            self.sio.connect(self.socketio_server)
            print(f"Connected to Socket.IO server at {self.socketio_server}")
        except Exception as e:
            print(f"Error connecting to Socket.IO server: {e}")

    def send_chunked(self, conn, data):
        try:
            send_json_data = json.dumps(data).encode()

            size = len(send_json_data)
            conn.sendall(struct.pack('!I', size))
            conn.sendall(send_json_data)  # Send all at once

        except Exception as e:
            print(f"Error in send_chunked: {e}")
            raise

    def receive_chunked(self, conn):
        try:
            size_data = conn.recv(4)
            if not size_data:
                return None
            total_size = struct.unpack('!I', size_data)[0]
            data = conn.recv(total_size)  # Receive all at once
            if not data:
                return None

            print(data)

            received_data = json.loads(data)
            return received_data

        except Exception as e:
            raise
            print(f"Error in receive_chunked: {e}")
            return None

    def send_output(self, mode, content):
        mode = {"document": "gather_case_details", "chat": "chat", "create": "gather_case_details"}[mode]
        try:
            self.sio.emit(mode, {"session_id": self.session_id, "text": content})
            print(f"Sent event '{mode}' with content to Socket.IO server")
        except Exception as e:
            print(f"Error sending event to Socket.IO server: {e}")

    def load_model(self, model_name):
        try:
            self.model = whisper.load_model(model_name)
            self.model_name = model_name
            return {"status": "success", "message": f"Loaded {model_name} model successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def receive_audio_data(self, conn):
        """Receive audio data in chunks"""
        try:
            message = self.receive_chunked(conn)
            if not message:
                return None, None

            audio_data = message.get("audio_data")
            mode = message.get("mode", "document")
            return audio_data, mode

        except Exception as e:
            print(f"Error receiving audio data: {e}")
            raise

    def detect_language(self, audio_data):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(base64.b64decode(audio_data))
                temp_path = temp_file.name

            audio = whisper.load_audio(temp_path)
            audio = whisper.pad_or_trim(audio)
            if self.model_name in ["large", "turbo"]:
                mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.model.device)
            else:
                mel = whisper.log_mel_spectrogram(audio, n_mels=80).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

            os.unlink(temp_path)

            return {"status": "success", "language": detected_lang}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def transcribe_audio(self, audio_data, mode):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(base64.b64decode(audio_data))
                temp_path = temp_file.name

            if mode == "create":
                hash_value = sha512(audio_data.encode()).hexdigest()
                self.session_id = self.file_hash_map.get(hash_value, str(uuid.uuid4()))
                self.file_hash_map[hash_value] = self.session_id

            audio = whisper.load_audio(temp_path)
            audio = whisper.pad_or_trim(audio)
            if self.model_name in ["large", "turbo"]:
                mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.model.device)
            else:
                mel = whisper.log_mel_spectrogram(audio, n_mels=80).to(self.model.device)
            options = whisper.DecodingOptions()
            result = whisper.decode(self.model, mel, options)

            os.unlink(temp_path)

            self.send_output(mode, result.text)
            return {"status": "success", "text": result.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def translate_audio(self, audio_data, mode):
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(base64.b64decode(audio_data))
                temp_path = temp_file.name

            if mode == "create":
                hash_value = sha512(audio_data.encode()).hexdigest()
                self.session_id = self.file_hash_map.get(hash_value, str(uuid.uuid4()))
                self.file_hash_map[hash_value] = self.session_id

            audio = whisper.load_audio(temp_path)
            audio = whisper.pad_or_trim(audio)
            if self.model_name in ["large", "turbo"]:
                mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.model.device)
            else:
                mel = whisper.log_mel_spectrogram(audio, n_mels=80).to(self.model.device)
            options = whisper.DecodingOptions(task="translate")
            result = whisper.decode(self.model, mel, options)

            os.unlink(temp_path)

            self.send_output(mode, result.text)
            return {"status": "success", "text": result.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def handle_client(self, conn, addr):
        print(f"New connection from {addr}")

        while True:
            try:
                message = self.receive_chunked(conn)
                if not message:
                    break

                command = message.get("command")

                if command == "load_model":
                    response = self.load_model(message["model_name"])
                elif command == "detect_language":
                    audio_data = message["audio_data"]
                    response = self.detect_language(audio_data)
                elif command == "transcribe":
                    audio_data = message["audio_data"]
                    mode = message.get("mode", "document")
                    response = self.transcribe_audio(audio_data, mode)
                elif command == "translate":
                    audio_data = message["audio_data"]
                    mode = message.get("mode", "document")
                    response = self.translate_audio(audio_data, mode)
                else:
                    response = {"status": "error", "message": "Unknown command"}

                self.send_chunked(conn, response)

            except Exception as e:
                raise
                print(f"Error handling client: {e}")
                break

        conn.close()
        print(f"Connection closed from {addr}")

    def start_tcp_server(self):
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
            mongo_client = MongoClient("mongodb+srv://Gilgamesh:mDJRw2rvTw3wbo5v@botdbcluster.4vxflu6.mongodb.net/")
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
