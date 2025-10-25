import serial
import time
import os
import sys
from io import BytesIO
from PIL import Image

CHUNK_SIZE = 200  # Payload bytes per chunk
HEADER_SIZE = 7   # Type(1) + ChunkID(2) + TotalChunks(2) + PayloadSize(2)
SERIAL_PORT = 'COM6'  # Nano transmitter
BAUD_RATE = 115200
MAX_RETRIES = 3   # Per chunk retries
RESET_DELAY = 5   # Increased for Arduino init
COMPRESSION_QUALITY = 85  # JPEG quality (0-100); lower = more compression

def transmit_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return False
    
    # Load and compress image
    with Image.open(image_path) as img:
        # Convert to RGB if needed (for PNG/JPG handling)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=COMPRESSION_QUALITY, optimize=True)
        image_data = buffer.getvalue()
    
    original_size = os.path.getsize(image_path)
    compressed_size = len(image_data)
    print(f"Original size: {original_size} bytes, Compressed size: {compressed_size} bytes (reduction: {((original_size - compressed_size) / original_size) * 100:.2f}%)")
    
    total_chunks = (len(image_data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Total chunks: {total_chunks}")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10)
        time.sleep(RESET_DELAY)  # Wait for Arduino
        ser.reset_input_buffer()  # Clear any garbage
        
        all_success = True
        for chunk_id in range(total_chunks):
            start = chunk_id * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(image_data))
            payload = image_data[start:end]
            payload_size = len(payload)
            
            packet = bytearray()
            packet.append(0x01)
            packet.extend(chunk_id.to_bytes(2, 'big'))
            packet.extend(total_chunks.to_bytes(2, 'big'))
            packet.extend(payload_size.to_bytes(2, 'big'))
            packet.extend(payload)
            
            success = False
            for retry in range(MAX_RETRIES):
                ser.write(len(packet).to_bytes(2, 'big'))
                ser.write(packet)
                ser.flush()
                
                response = ser.read(1)
                if response == b'\x01':
                    success = True
                    print(f"Chunk {chunk_id+1}/{total_chunks} sent successfully")
                    break
                elif response:
                    print(f"Chunk {chunk_id+1} failed (response: {response}), retry {retry+1}")
                else:
                    print(f"Chunk {chunk_id+1} timeout, retry {retry+1}")
                time.sleep(1)
            
            if not success:
                print(f"Failed to send chunk {chunk_id+1} after {MAX_RETRIES} retries")
                all_success = False
                break
        
        ser.close()
        print("Transmission complete" if all_success else "Transmission incomplete")
        return all_success
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transmit_compressed.py <image_name.jpg>")
        sys.exit(1)
    image_name = sys.argv[1]
    transmit_image(image_name)