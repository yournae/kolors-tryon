# Kolors Virtual Try-On CLI 👕

CLI tool for [Kwai-Kolors/Kolors-Virtual-Try-On](https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On) HuggingFace Space.

Upload a person photo + garment photo → get AI-generated try-on result.

## Install

### 1. Python 3

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y python3 python3-pip python3-venv

# Arch
sudo pacman -S python python-pip

# macOS
brew install python3

# Windows — download installer dari https://www.python.org/downloads/
# centang "Add Python to PATH" saat install
```

Verifikasi:
```bash
python3 --version   # minimal 3.9+
```

### 2. Dependencies

```bash
# (opsional) bikin venv dulu biar rapi
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Setup

### 1. Environment Variables (Photo-to-Video)

Photo-to-Video butuh HF token. Copy `.env.example` ke `.env` lalu isi:

```bash
cp .env.example .env
# Edit .env, isi HF_TOKENS dengan token dari https://huggingface.co/settings/tokens
```

Bisa single token atau comma-separated (untuk rotasi):
```
HF_TOKENS=hf_xxxxxxx,hf_yyyyyyy,hf_zzzzzzz
```

### 2. Folder Structure

```
kolors-tryon/
├── tryon.py          # main script
├── requirements.txt
├── README.md
├── models/           # taruh foto orang di sini
│   └── README.md
├── garments/         # taruh foto baju di sini
│   └── README.md
└── results/          # hasil try-on otomatis tersimpan di sini
    └── README.md
```

## Usage

### Interactive Menu (default)

```bash
python3 tryon.py
```

Muncul menu bertingkat navigasi pakai panah keyboard:

```
╔═══════════════════════════════════════════════╗
║       👕  Kolors Virtual Try-On CLI  👕      ║
╚═══════════════════════════════════════════════╝

  Main Menu:
  > 🔥  Try On           (models: 2, garments: 3)
    📁  Lihat Files      (results: 5)
    ⚙️   Settings         (seed: 42, timeout: 180s)
    🚪  Exit
```

**Try On sub-menu:**

```
  Try-On Menu:
  > 📷  Pilih Model    → model.png
    👔  Pilih Garment  → Blouse.webp
    🌱  Set Seed       → 42
    🎲  Random Seed    → OFF
    ─────────────────────────────
    ▶️   RUN Try-On
    ←  Kembali ke Menu Utama
```

**Alur:**
1. Pilih **Pilih Model** → pilih foto dari `models/`
2. Pilih **Pilih Garment** → pilih foto dari `garments/`
3. Set seed (opsional) atau toggle Random Seed
4. Pilih **RUN Try-On** → proses berjalan, hasil masuk `results/`

### Direct CLI Mode

```bash
# Basic
python3 tryon.py --cli -p person.jpg -g shirt.jpg -o result.png

# Atau langsung (auto-detect dari -p flag)
python3 tryon.py -p person.jpg -g shirt.jpg -o result.png

# With seed
python3 tryon.py -p person.jpg -g dress.jpg --seed 12345 -o output.png

# Random seed
python3 tryon.py -p person.jpg -g shirt.jpg --random-seed -v

# List local images
python3 tryon.py --cli --list
```

## Options (CLI mode)

| Flag | Short | Description |
|------|-------|-------------|
| `--person` | `-p` | Person image (file or URL) |
| `--garment` | `-g` | Garment image (file or URL) |
| `--output` | `-o` | Output path (default: `results/result.png`) |
| `--seed` | `-s` | Seed 0-999999 (default: 42) |
| `--random-seed` | | Use random seed |
| `--timeout` | `-t` | Timeout in seconds (default: 180) |
| `--verbose` | `-v` | Show progress |
| `--list` | `-l` | List local images |
| `--cli` | | Force CLI mode (skip interactive menu) |

## Python API

```python
from tryon import submit_tryon

result = submit_tryon(
    person_path="models/me.jpg",
    garment_path="garments/shirt.jpg",
    seed=42,
    verbose=True,
    output_path="results/output.png",
)
# Returns: {"path": "...", "seed": 42, "elapsed": 28.5, "size": 9962, ...}
```

## Tips

- Taruh foto model di `models/`, foto baju di `garments/`
- Gunakan foto orang yang jelas, menghadap depan
- Background polos menghasilkan hasil lebih baik
- Free tier HF → 15-60s per request, tergantung queue
- Output selalu WebP (400×500 px) meskipun ekstensi .png
