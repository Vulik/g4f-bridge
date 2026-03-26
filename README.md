# g4f-Bridge

**OpenAI-Compatible API Bridge for g4f (GPT4Free)**

g4f-Bridge adalah middleware yang menghubungkan aplikasi OpenAI-compatible (seperti PicoClaw) dengan g4f tanpa modifikasi apapun pada aplikasi client.

## Fitur

- ✅ **100% OpenAI API Compatible** - PicoClaw/client tidak tahu bedanya
- ✅ **Auto Provider Discovery** - Scan otomatis semua provider g4f
- ✅ **Smart Routing** - Fallback otomatis jika provider gagal
- ✅ **Circuit Breaker** - Proteksi dari provider yang error
- ✅ **Rate Limiting** - Prevent abuse
- ✅ **Token Management** - Session rotation untuk unlimited tokens
- ✅ **Streaming Support** - Real-time response
- ✅ **Auto Update** - Update g4f otomatis
- ✅ **Termux Ready** - Berjalan di Android via Termux
- ✅ **Low Memory** - Target < 100MB RAM

## Quick Start

### Termux (Android)

```bash
# 1. Install Termux dari F-Droid

# 2. Download & jalankan setup
curl -fsSL https://raw.githubusercontent.com/.../setup-termux.sh | bash

# Atau manual:
pkg install python git
git clone https://github.com/.../g4f-bridge.git
cd g4f-bridge
bash setup-termux.sh
```

### Linux

```bash
# 1. Clone repository
git clone https://github.com/.../g4f-bridge.git
cd g4f-bridge

# 2. Jalankan setup
bash setup.sh
```

### Windows

```batch
REM 1. Clone repository
git clone https://github.com/.../g4f-bridge.git
cd g4f-bridge

REM 2. Jalankan setup
setup.bat
```

## Konfigurasi PicoClaw

**PENTING: Tidak ada modifikasi pada PicoClaw!**

PicoClaw sudah support custom OpenAI endpoint secara default. Anda hanya perlu mengubah setting:

1. Buka setting PicoClaw
2. Set **Base URL**: `http://localhost:8080/v1` (atau IP bridge Anda)
3. Set **API Key**: (lihat saat bridge start, atau cek `/api-key`)
4. Done! PicoClaw akan menggunakan g4f melalui bridge

```
PicoClaw Settings:
┌─────────────────────────────────────┐
│ API Configuration                   │
├─────────────────────────────────────┤
│ Base URL: http://localhost:8080/v1  │
│ API Key:  sk-xxxxxxxxxxxxxxxxxxxxx  │
│ Model:    gpt-4 (atau auto)         │
└─────────────────────────────────────┘
```

## Cara Menjalankan

### Manual

```bash
cd ~/g4f-bridge
source venv/bin/activate  # Linux/Mac/Termux
# atau: venv\Scripts\activate  # Windows

python main.py
```

### Menggunakan Script

```bash
# Linux/Termux
./start.sh

# Windows
start.bat
```

### Systemd Service (Linux)

```bash
sudo cp g4f-bridge.service /etc/systemd/system/
sudo systemctl enable g4f-bridge
sudo systemctl start g4f-bridge
```

## Endpoints

### OpenAI-Compatible (dipakai PicoClaw)

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/models` | GET | List models |
| `/v1/models/{id}` | GET | Get model info |

### Management (tidak dipakai PicoClaw)

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/health` | GET | Health check |
| `/status` | GET | Bridge status |
| `/providers` | GET | List providers |
| `/compatibility` | GET | Model-provider map |
| `/scan` | POST | Trigger rescan |
| `/update` | POST | Update g4f |
| `/version` | GET | Version info |
| `/api-key` | GET | Get API key |
| `/config` | GET | Current config |

## Testing dengan curl

```bash
# Get API key
curl http://localhost:8080/api-key

# Health check
curl http://localhost:8080/health

# List models
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/v1/models

# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4",
       "messages": [
         {"role": "user", "content": "Hello!"}
       ]
     }'

# Streaming
curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4",
       "messages": [{"role": "user", "content": "Tell me a joke"}],
       "stream": true
     }'
```

## Konfigurasi

File konfigurasi: `~/g4f-bridge/config/config.json`

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "api_key": "auto-generated"
  },
  "g4f": {
    "default_model": "auto",
    "preferred_models": ["gpt-4o", "gpt-4", "claude-3.5-sonnet"],
    "timeout_seconds": 60
  },
  "circuit_breaker": {
    "failure_threshold": 3,
    "cooldown_seconds": 600
  }
}
```

## Troubleshooting

### "No providers available"

```bash
# Coba rescan providers
curl -X POST http://localhost:8080/scan

# Atau restart bridge
```

### "tiktoken gagal install" (Termux)

Ini normal. Bridge akan menggunakan fallback token counter.

### "Connection refused"

1. Pastikan bridge berjalan: `ps aux | grep main.py`
2. Cek port tidak dipakai: `netstat -tlnp | grep 8080`
3. Cek firewall jika akses dari device lain

### "Invalid API key"

1. Cek API key: `curl http://localhost:8080/api-key`
2. Pastikan header benar: `Authorization: Bearer YOUR_KEY`

### Provider sering error

```bash
# Cek status providers
curl http://localhost:8080/providers

# Update g4f ke versi terbaru
curl -X POST http://localhost:8080/update
```

### High memory usage

Edit `config.json`:
```json
{
  "token_manager": {
    "sliding_window_messages": 10
  },
  "scanner": {
    "scan_interval_minutes": 60
  }
}
```

## Arsitektur

```
┌─────────────────────┐
│      PicoClaw       │  (TIDAK DIMODIFIKASI)
│  base_url → Bridge  │
└─────────┬───────────┘
          │ HTTP (OpenAI format)
          ▼
┌─────────────────────┐
│    g4f-Bridge       │
│  ├─ API Layer       │
│  ├─ Router          │
│  ├─ Scanner         │
│  ├─ Token Manager   │
│  └─ Resilience      │
└─────────┬───────────┘
          │ Python call
          ▼
┌─────────────────────┐
│        g4f          │
│   (GPT4Free lib)    │
└─────────────────────┘
```

## License

MIT License

## Credits

- [g4f (GPT4Free)](https://github.com/xtekky/gpt4free)
- [PicoClaw](https://github.com/sipeed/picoclaw)


---

## Panduan Penggunaan

### 1. Setup dari Nol

#### Termux (Android):

```bash
# Install Termux dari F-Droid (BUKAN Play Store)
# Buka Termux, lalu:

pkg update && pkg upgrade -y
pkg install git python -y

# Clone atau buat folder
mkdir -p ~/g4f-bridge && cd ~/g4f-bridge

# Copy semua file Python ke folder ini
# Kemudian jalankan setup:
bash setup-termux.sh
```

#### Linux:

```bash
# Clone atau buat folder
mkdir -p ~/g4f-bridge && cd ~/g4f-bridge

# Copy semua file Python ke folder ini
# Kemudian:
bash setup.sh
```

#### Windows:

```batch
REM Buat folder C:\Users\<you>\g4f-bridge
REM Copy semua file ke folder tersebut
REM Jalankan setup.bat
```

### 2. Arahkan PicoClaw ke Bridge

**Tidak perlu modifikasi PicoClaw sama sekali!**

PicoClaw sudah memiliki fitur untuk custom API endpoint. Anda hanya perlu:

1. Jalankan bridge: `python main.py`
2. Catat API key yang muncul di console
3. Di PicoClaw, set:
   - **Base URL**: `http://<IP-BRIDGE>:8080/v1`
   - **API Key**: `<key dari console>`
   - **Model**: `gpt-4` atau `auto`

Jika PicoClaw di device yang sama dengan bridge:
- Base URL: `http://localhost:8080/v1`

Jika berbeda device (misal bridge di PC, PicoClaw di HP):
- Base URL: `http://192.168.1.xxx:8080/v1` (IP PC Anda)

### 3. Test dengan curl

```bash
# Dapatkan API key
API_KEY=$(curl -s http://localhost:8080/api-key | grep -o '"api_key":"[^"]*' | cut -d'"' -f4)
echo "API Key: $API_KEY"

# Test health
curl http://localhost:8080/health

# Test models
curl -H "Authorization: Bearer $API_KEY" http://localhost:8080/v1/models

# Test chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Say hello!"}]
  }'

# Test streaming
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "stream": true
  }'
```

### 4. Troubleshooting Umum

| Masalah | Solusi |
|---------|--------|
| `No providers available` | Jalankan `curl -X POST http://localhost:8080/scan` untuk rescan |
| `Connection refused` | Pastikan bridge berjalan dan port 8080 tidak diblok firewall |
| `Invalid API key` | Cek key dengan `curl http://localhost:8080/api-key` |
| `tiktoken error` di Termux | Normal, bridge otomatis pakai fallback |
| Memory tinggi | Kurangi `sliding_window_messages` di config |
| Semua provider gagal | Update g4f: `curl -X POST http://localhost:8080/update` |
| Bridge crash | Cek log di `~/g4f-bridge/data/logs/bridge.log` |

### 5. Tips Termux

```bash
# Agar tidak di-kill Android saat layar mati:
termux-wake-lock

# Jalankan di background:
nohup python main.py > bridge.log 2>&1 &

# Cek jalan atau tidak:
ps aux | grep main.py

# Stop:
pkill -f main.py
```