# ReelMotion AI

**Local-first · Open-source · Zero API credits**

AI Character Replacement & Reels Recording Studio

Upload character reference images. Record yourself with your webcam. ReelMotion AI
transfers your body movements, facial expressions, and speech to the selected character,
preserves your original background, and exports a vertical video for social media.

## ⚠️ Project Status

**Current Focus: Neural AI Feasibility & Hardware Grounding**

| Phase | Description | Status |
|:---:|:---|:---:|
| 1 | Project structure, backend foundation & APIs | 🟡 Implemented (pending full integration verification) |
| 2 | Camera + microphone + recording | 🟡 Basic functionality implemented |
| 3 | Person tracking + segmentation | 🟡 OpenCV fallback active; neural tracking pending |
| 4 | Character reference manager | 🟡 Basic validation implemented |
| 5 | AI character replacement (genuine neural model) | 🔴 **NOT IMPLEMENTED as genuine neural generation** |
| 6 | Lip synchronization (genuine neural model) | 🔴 **NOT IMPLEMENTED as genuine neural lip sync** |
| 7 | Compositing + background restoration | 🟡 Basic functionality implemented |
| 8 | 1080×1920 MP4 export | 🟡 Basic functionality implemented |
| 9 | Testing + validation | 🟡 74 unit tests passing (scaffolding/fallbacks verified) |
| 10 | Packaging + documentation | 🟡 Prototype stage |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 4 GB VRAM | NVIDIA with 8+ GB VRAM |
| RAM | 16 GB | 24+ GB |
| OS | Windows 10/11 | Windows 11 |
| Storage | 20 GB free | 50+ GB free |
| Python | 3.11+ | 3.12 |
| Node.js | 18+ | 20+ LTS |

## Quick Start

### Prerequisites

- Python 3.11+ with pip
- Node.js 18+ with npm
- NVIDIA GPU with CUDA drivers (optional for Phase 1)

### Installation

```powershell
# Clone the repository
git clone <repo-url>
cd reelmotion-ai

# Create Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -e ".[dev]"

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running

```powershell
# Start both backend and frontend
.\scripts\start.ps1

# Or start separately:
# Backend
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Frontend
cd frontend
npm run dev
```

### Testing

```powershell
# Run all tests
.\scripts\test.ps1

# Backend tests only
pytest tests/backend/ -v

# With coverage
pytest tests/backend/ -v --cov=backend
```

## Architecture

```
User Browser (React)
    ├── Camera preview (MediaStream API)
    ├── Audio capture (MediaStream API)
    └── WebSocket ←→ Backend status/telemetry
            │
      FastAPI Backend
    ├── Hardware detection
    ├── Model management
    ├── Character validation
    ├── Pose/face/hand tracking
    ├── AI generation engine
    ├── Video compositing
    └── FFmpeg export
```

**Key design decision:** Camera preview and raw recording happen in the browser
using MediaStream. WebSocket is used for backend communication, hardware
monitoring, job progress, and processing events — NOT for streaming raw video
frames.

## Project Structure

```
reelmotion-ai/
├── frontend/          # React + Vite UI
├── backend/           # Python FastAPI server
│   ├── api/           # REST + WebSocket endpoints
│   ├── core/          # Config, logging, shared utilities
│   ├── capture/       # Camera + audio capture (Phase 2)
│   ├── vision/        # Pose, face, hand tracking (Phase 3)
│   ├── character/     # Character reference management (Phase 4)
│   ├── generation/    # AI model backends (Phase 5)
│   ├── rendering/     # Compositing + export (Phase 7-8)
│   └── services/      # Hardware, models, job queue
├── models/            # AI model weights (NOT in git)
├── characters/        # User character references
├── recordings/        # Raw camera recordings
├── exports/           # Final exported videos
├── tests/             # Automated tests
├── configs/           # YAML configuration files
└── scripts/           # Startup + utility scripts
```

## Cost

| Item | Cost |
|------|------|
| Software | ₹0 |
| API credits | ₹0 |
| Cloud inference | ₹0 |
| Subscription | ₹0 |

All AI inference runs locally on your machine.

## License

MIT License — see [LICENSE](LICENSE) for details.

**Important:** Individual AI model weights have their own licenses. Check
`configs/model_manifest.yaml` for per-model license information before
commercial use.

