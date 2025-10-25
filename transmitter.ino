#include <SPI.h>
#include <LoRa.h>

#define LORA_FREQ 868E6  // 868 MHz
#define LORA_SF 7
#define LORA_BW 125E3
#define LORA_CR 5
#define SYNC_WORD 0x34
#define DEBUG false      // Set to true for debug prints

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  if (!LoRa.begin(LORA_FREQ)) {
    if (DEBUG) Serial.println("LoRa init failed!");
    while (1);
  }
  
  LoRa.setSpreadingFactor(LORA_SF);
  LoRa.setSignalBandwidth(LORA_BW);
  LoRa.setCodingRate4(LORA_CR);
  LoRa.setSyncWord(SYNC_WORD);
  LoRa.enableCrc();
  
  if (DEBUG) Serial.println("Receiver ready");
}

void loop() {
  int pktSize = LoRa.parsePacket();
  if (pktSize >= 7) {
    uint8_t packet[pktSize];
    LoRa.readBytes(packet, pktSize);
    
    if (packet[0] == 0x01) {
      uint16_t chunkId = (packet[1] << 8) | packet[2];
      uint16_t totalChunks = (packet[3] << 8) | packet[4];
      uint16_t payloadSize = (packet[5] << 8) | packet[6];
      
      if (payloadSize != pktSize - 7) {
        if (DEBUG) Serial.println("Error: Payload size mismatch");
        return;
      }
      
      if (DEBUG) {
        Serial.print("Received chunk "); Serial.print(chunkId); Serial.print(" of "); Serial.println(totalChunks);
      }
      
      Serial.write((uint8_t)(pktSize >> 8));
      Serial.write((uint8_t)pktSize);
      Serial.write(packet, pktSize);
      
      LoRa.beginPacket();
      LoRa.write(0x02);
      LoRa.write((uint8_t)(chunkId >> 8));
      LoRa.write((uint8_t)chunkId);
      LoRa.write(0x00);
      LoRa.write(0x00);
      LoRa.endPacket();
      if (DEBUG) { Serial.print("ACK sent for chunk "); Serial.println(chunkId); }
    }
  }
}