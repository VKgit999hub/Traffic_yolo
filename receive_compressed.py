import serial
import time
import os
from collections import defaultdict

HEADER_SIZE = 7   # Type(1) + ChunkID(2) + TotalChunks(2) + PayloadSize(2)
SERIAL_PORT = 'COM6'  # Uno receiver
BAUD_RATE = 115200
OUTPUT_IMAGE = "received_image.jpg"  # Updated to generic name since input varies
RESET_DELAY = 5   # Increased for Arduino init

def receive_image():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10)
        time.sleep(RESET_DELAY)
        ser.reset_input_buffer()  # Clear garbage
        
        chunks = {}
        total_chunks = None
        received_count = 0
        
        print("Waiting for chunks...")
        while True:
            if ser.in_waiting >= 2:
                size_bytes = ser.read(2)
                if len(size_bytes) != 2:
                    print("Error: Incomplete size read")
                    continue
                packet_size = (size_bytes[0] << 8) | size_bytes[1]
                
                if packet_size < HEADER_SIZE or packet_size > 255:
                    print(f"Error: Invalid packet size ({packet_size})")
                    continue
                
                packet = ser.read(packet_size)
                if len(packet) != packet_size:
                    print("Error: Incomplete packet read")
                    continue
                
                if packet[0] != 0x01:
                    print(f"Error: Not a data packet (type: {packet[0]:02x})")
                    continue
                
                chunk_id = (packet[1] << 8) | packet[2]
                current_total = (packet[3] << 8) | packet[4]
                payload_size = (packet[5] << 8) | packet[6]
                
                if payload_size != packet_size - HEADER_SIZE:
                    print("Error: Payload size mismatch")
                    continue
                
                payload = packet[HEADER_SIZE:]
                
                if total_chunks is None:
                    total_chunks = current_total
                elif total_chunks != current_total:
                    print("Error: Total chunks mismatch")
                    continue
                
                if chunk_id not in chunks:
                    chunks[chunk_id] = payload
                    received_count += 1
                    print(f"Received chunk {chunk_id+1}/{total_chunks}")
                
                if received_count == total_chunks:
                    print("All chunks received. Reassembling image...")
                    image_data = b''.join(chunks[i] for i in sorted(chunks))
                    
                    with open(OUTPUT_IMAGE, 'wb') as f:
                        f.write(image_data)
                    print(f"Image saved to {OUTPUT_IMAGE}")
                    break
            
            time.sleep(0.1)
        
        ser.close()
    except serial.SerialException as e:
        print(f"Serial error: {e}")

if __name__ == "__main__":
    receive_image()