---
title: "YOLOv8 and IoT Agricultural Pest Detection System"
excerpt: "Undergraduate thesis combining embedded sensing, live video, and YOLOv8 pest detection."
collection: portfolio
date: 2026-01-01
---

- Designed the embedded control architecture on an STM32F103C8T6 microcontroller, integrating DHT22 temperature and humidity sensing with an effective accumulated-temperature algorithm for automated SMS pest-outbreak alerts.
- Configured an ESP32-S3-WROOM-1U WiFi module to stream live field video from the microcontroller node to a host-computer gateway for real-time processing.
- Built the video ingestion and preprocessing pipeline with OpenCV to feed live field footage into the inference pipeline.
- Fine-tuned and deployed a YOLOv8x model for real-time pest detection on the IP102 dataset, with results visualized through a PyQt5 interface backed by SQLite.
