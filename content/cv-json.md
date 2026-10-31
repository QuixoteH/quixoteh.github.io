## Hai Huang

**M.Sc. Student in Robotics and Intelligent Systems**

[quixotehh@gmail.com](mailto:quixotehh@gmail.com)\
<https://quixoteh.github.io>\
Singapore, SG\
GitHub: [QuixoteH](https://github.com/QuixoteH)\
LinkedIn: [quixoteh](https://www.linkedin.com/in/quixoteh)

## Education

### M.S. in Robotics and Intelligent Systems

**Master of Science** · 08/2026 – Present\
Nanyang Technological University, School of Mechanical and Aerospace Engineering

### B.E. in Internet of Things

**Bachelor of Engineering** · 09/2022 – 06/2026\
Northeast Agricultural University, College of Intelligent Science and Engineering\
GPA: 85/100\
Courses: Data Structures and Algorithms, Microcontroller Development, Principles of Sensor Networks

## Work Experience

### Research Intern

**Singapore Institute of Manufacturing Technology (SIMTech), A\*STAR** · 08/2026 – Present\
Singapore

Research on force-aware robot learning and robot manipulation.

### Solutions Engineer Intern

**China Unicom Chengdu Branch, Digital Technology Center** · 07/2025 – 09/2025

Supported AI solution planning and technical delivery for government and enterprise projects.

- Developed an automation plan for probation recommendation reports using China Unicom's in-house UniAI LLM, following parameter verification and delivery workflows.
- Prepared technical implementation plans and equipment specifications for the Chengdu Comprehensive Disaster-Prevention Capacity Enhancement Project.
- Contributed to the Blossom Cup 5G Competition proposal for the FormPass 5G+AI Government Service System.

## Skills

### Programming

- Python
- C/C++

### Software and Tools

- ROS/ROS2
- Git
- PyTorch
- OpenCV
- MuJoCo
- ManiSkill3

### Hardware

- STM32
- ESP32
- SO-101 Robot Arm

## Publications

### Synergistic Modality-and-Slice Memory Framework for Cross-Modal 3D Brain Tumor Segmentation

**IEEE Transactions on Multimedia** · 10/2026\
Published, pp. 1–12. DOI: 10.1109/TMM.2026.3739470\
Yuxiang Luo, Qing Xu, **Hai Huang**, Yuqi Ouyang, Xiangjian He, Zhen Chen, Wenting Duan, Jiebo Luo\
[IEEE Xplore](https://ieeexplore.ieee.org/document/11718973) · [DOI](https://doi.org/10.1109/TMM.2026.3739470) · [Code](https://github.com/QuixoteH/MSM-Seg)

## Research Projects

### SO-101 Real-World Robotic Learning with LeRobot

**Real-Robot Deployment** · 07/2026

- Collected 90 teleoperated episodes (17,778 frames at 20 Hz) with dual cameras across two block colors and three lighting conditions on the Huaner SO-101 arm.
- Fine-tuned a language-conditioned SmolVLA policy for 25,000 steps with LeRobot, modifying the source for multi-view RGB input and debugging calibration and inference failures.
- Reached 9/10 on consecutive real-robot demonstrations for the green block, while the purple block stayed at 5/10.

[Code](https://github.com/QuixoteH/SO-101-Real-World-Robotic-Learning-with-LeRobot)

### YOLOv8 and IoT Agricultural Pest Detection System

**Undergraduate Thesis** · 01/2026

- Designed STM32F103C8T6 firmware with DHT22 sampling, accumulated-temperature outbreak alerts over SMS, and ESP32 video streaming to a host computer.
- Fine-tuned YOLOv8x on a 7-class IP102 subset and built a PyQt5 application with SQLite logging for real-time pest detection.
- Reached 0.793 mAP@0.5 (precision 0.723, recall 0.815), choosing YOLOv8x after YOLOv8s performed poorly.

[Firmware Code](https://github.com/QuixoteH/STM32-PestMonitor) · [Host-Computer Code](https://github.com/QuixoteH/upper-computer)

### MSM-Seg: Cross-Modal 3D Brain Tumor Segmentation

**IEEE Transactions on Multimedia** · 03/2025

- Reimplemented nnU-Net, 3D-TransUNet, TransBTS, SIN2GR, UNETR++, and other baselines under one training setting on BraTS-METS and BraTS-AGPT from BraTS 2024.
- Measured Dice and HD95 on the ET, TC, and WT sub-regions to benchmark MSM-Seg, a modality-and-slice memory framework with category-agnostic prompting.
- Showed MSM-Seg reaching 79.51% and 83.84% average Dice, 2.04% and 1.36% above SAM-Med3D.

[Paper](https://ieeexplore.ieee.org/document/11718973) · [Code](https://github.com/QuixoteH/MSM-Seg)

## Competitions

### RoboSynChallenge @ NeurIPS 2026

**23rd · Preliminary Stage** · 10/2026\
Team LBWSquare

- Built a 10-task simulated manipulation submission by selecting, for each task, the best checkpoint among the official ACT, Diffusion Policy, and SmolVLA baselines, replacing it only after a challenger won paired same-seed tests.
- Trained pi0.5 for handle-basket and tuned replanning intervals, raising handle-basket success from 35% to 90% and water-pouring from 10/20 to 17/20.
- Ranked 23rd in the preliminary stage, only 2 places short of the 21 finalists, on hidden held-out parameters without leaderboard feedback.

[Leaderboard](https://robosyn-bench.net/#/leaderboard)

### Marso Hack Berlin 2026 – Robot Parcel Sorting Challenge

**Reproduction Study** · 07/2026

- Built State Diffusion Policy and RGB ACT training and evaluation pipelines for color-matched parcel sorting with a Franka Panda robot in ManiSkill3 and SAPIEN.
- Evaluated State Diffusion Policy over fixed 50-episode runs, obtaining sort accuracies of 0.390, 0.090, and 0.103 on Easy, Medium, and Hard (weighted 0.1565).
- Completed 30,000 offline updates for the Easy RGB ACT policy and verified checkpoint loading and submission packaging.

[Code](https://github.com/QuixoteH/Marso-Hack-Berlin-2026-Robot-Parcel-Sorting-Challeng) · [Competition Page](https://github.com/marso-robotics/berlin-marso-hackathon)

## Awards

### Silver Medal – Home Credit: Credit Risk Model Stability

**Kaggle Competition Award** · 05/2024\
Ranked in the top 0.5%.

## Languages

- Chinese: Native
- English: IELTS 7.0

## Interests

### Robot Learning and Manipulation

Force-aware Robot Learning, Vision-Language-Action Models
