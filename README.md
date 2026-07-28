# ViSTAR (Vietnamese Sign Language Translation and Recognition)
<div align="center">

[한국어]() | [日本語]() | [Русский]() | [Deutsch]() | [Français]() | [Español]() | [Português]() | [Türkçe]() | [Tiếng Việt]() | [العربية]()


<div>
    <img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation">
    <img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls">
    <img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue">
    <img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue">
    <img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue">
    <br>
    <img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
    <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle">
</div>

  <br>
</div>

![ViSTAR Logo](src/fe/static/css/VISTAR.webp)
A comprehensive system for real-time sign language translation using AI and computer vision. ViSTAR enables bidirectional translation between text and sign language through video frames.

## 🌟 Features

- **Text to Sign Language**: Convert written text into sign language video frames
- **Sign Language to Text**: Translate sign language gestures into written text
- **Real-time Processing**: WebSocket-based streaming for smooth real-time translations
- **User Management**: Secure authentication and user data management
- **Modern UI**: Responsive interface built with TailwindCSS

## 🏗️ Architecture

The project is structured into several key components:

- **Frontend (src/fe/)**: FastAPI-based web interface with WebSocket support
- **AI Service (src/ai/)**: Machine learning models for sign language processing
- **Backend (src/be/)**: Core business logic and data management
- **Streaming (src/streaming/)**: gRPC-based streaming service for real-time communication

## 🔧 Technologies

- **Backend Framework**: FastAPI, Django
- **AI/ML**: PyTorch, MediaPipe
- **Communication**: gRPC, WebSockets
- **Frontend**: TailwindCSS
- **Database**: SQLAlchemy
- **Search**: Elasticsearch
- **Authentication**: Passlib, bcrypt

## 📋 Prerequisites

- Python 3.x
- Node.js (for TailwindCSS)
- Docker (optional, for containerized deployment)

## 🚀 Installation

1. Clone the `final` branch:
```bash
git clone --branch final https://github.com/hoangtrungkien2109/ViSTAR.git
cd ViSTAR
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the frontend assets:
```bash
cd src/fe
npm install
```

## 💻 Development Setup

1. Start the streaming service:
```bash
python src/streaming/main.py
```

2. Launch the AI service:
```bash
python src/ai/main.py
```

3. Run the frontend server:
```bash
cd src/fe
python main.py
```

## 🐳 Docker Deployment

The project includes Docker support for both development and production environments:

Development:
```bash
docker-compose -f docker-compose.dev.yaml up
```

Production:
```bash
docker-compose -f docker-compose.prod.yaml up
```

## Runtime Sign Recognizer

The sign-to-text camera path supports two recognizers through one stable
interface:

| Value | Input | Model |
|---|---|---|
| `baseline` | Existing 40-frame, 457-D ViSTAR feature sequence | Original ViSTAR Transformer |
| `felf_slr` | Raw normalized landmarks converted to `[T,165]`, `[T,165]`, and `[T,23]` | Baseline/LRG/RF logit fusion, optionally followed by canonical MT fusion |

Copy `.env.example` to `.env` and choose the active recognizer:

```dotenv
SIGN_RECOGNIZER=baseline
BASELINE_CHECKPOINT=src/be/n2_dict.pth
```

For the complete FELF-SLR path:

```dotenv
SIGN_RECOGNIZER=felf_slr
RECOGNIZER_LABELS_FILE=/absolute/path/to/vistar_labels.json
FELF_CHECKPOINT=/absolute/path/to/vistar_felf_stage1.pth
FELF_MT_CHECKPOINT=/absolute/path/to/vistar_mt.pth
FELF_REQUIRE_MT=true
```

The FELF checkpoints must be trained for exactly the same labels and class
order as `RECOGNIZER_LABELS_FILE`. WLASL checkpoints cannot be used directly
for the 19-class Vietnamese pilot. Research checkpoints are intentionally not
committed to this application repository.

At startup, ViSTAR validates checkpoint compatibility and fails clearly when a
selected recognizer is not configured. It never silently falls back to another
model. The active non-secret configuration is available at:

```text
GET /recognizer/status
```

The canonical fusion defaults are:

```text
Stage 1: (Baseline + 1.0*LRG + 0.75*RF) / 2.75
Stage 2: 0.5*Baseline + 1.0*Stage1 + 0.5*MT
```

See `.env.example` for separate expert-checkpoint options and architecture
settings. Configuration values must match those used to train the checkpoint.

## Environment Variables

Never commit a real `.env`. Start from the safe template:

```bash
cp .env.example .env
```

The template covers streaming, dictionary, Baseline, FELF-SLR, MorphTraj, and
device settings. It also disables automatic admin creation by default. For a
private development deployment, set `CREATE_DEFAULT_ADMIN=true` and provide
all three `DEFAULT_ADMIN_*` values; never commit those values.

## 📁 Project Structure

```
ViSTAR/
├── src/
│   ├── ai/                 # AI/ML models and services
│   ├── be/                 # Backend services
│   │   └── recognizers/    # Runtime Baseline/FELF-SLR registry
│   ├── fe/                 # Frontend application
│   ├── streaming/          # gRPC streaming services
│   └── init_data/         # Initial data and setup scripts
├── data/                  # Data storage
├── docker-compose.*.yaml  # Docker configurations
└── requirements.txt       # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
