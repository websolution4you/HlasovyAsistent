# Gemma 4 Voice Assistant - Setup Guide

Tento projekt implementuje FastAPI backend pre Twilio Media Streams s využitím **Gemma 4 (E4B/E2B)** modelu pre natívne audio-to-audio spracovanie. Žiadne medzikroky s Whisperom alebo TTS - model priamo spracováva audio vstup a generuje audio výstup.

## 🎯 Hlavné vlastnosti

- ✅ **Natívne Audio-to-Audio**: Gemma 4 priamo spracováva audio bez STT/TTS
- ✅ **Nízka latencia**: Optimalizované pre real-time konverzácie
- ✅ **Barge-in support**: Používateľ môže prerušiť odpoveď modelu
- ✅ **Lokálne GPU**: Beží kompletne na vašej GPU (žiadne API volania)
- ✅ **Slovenčina**: Nastavené pre prirodzenú slovenskú konverzáciu
- ✅ **Function calling**: Podpora pre overenie adresy, uloženie objednávky, atď.

## 📋 Požiadavky

### Hardware
- **GPU**: NVIDIA GPU s minimálne 16GB VRAM (odporúčané 24GB+)
- **CUDA**: CUDA 11.8 alebo novšia
- **RAM**: Minimálne 32GB systémovej RAM

### Software
- Python 3.10+
- CUDA Toolkit
- Git

## 🚀 Inštalácia

### 1. Klonuj repozitár a prepni na Gemma4 branch

```bash
git checkout Gemma4
```

### 2. Vytvor virtuálne prostredie

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Nainštaluj závislosti

```bash
pip install -r requirements_gemma4.txt
```

**Poznámka**: vLLM automaticky nainštaluje správnu verziu PyTorch s CUDA supportom. Ak máš problémy, môžeš manuálne nainštalovať PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Stiahni Gemma 4 model

Gemma 4 modely sú dostupné cez Hugging Face. Potrebuješ prístupový token:

```bash
# Prihlás sa do Hugging Face
huggingface-cli login

# Stiahni model (E4B variant - 4B parametrov)
huggingface-cli download google/gemma-4-e4b --local-dir ./models/gemma-4-e4b

# Alebo E2B variant (2B parametrov - menší, rýchlejší)
# huggingface-cli download google/gemma-4-e2b --local-dir ./models/gemma-4-e2b
```

**Alternatíva**: Ak chceš použiť google-ai-edge SDK namiesto vLLM:
1. Odkomentuj `google-ai-edge` v `requirements_gemma4.txt`
2. Nastav `USE_VLLM=false` v `.env`

## ⚙️ Konfigurácia

### 1. Vytvor `.env` soubor

```bash
cp .env.example .env
```

### 2. Nastav environment premenné

```env
# Gemma 4 konfigurácia
GEMMA4_MODEL_PATH=./models/gemma-4-e4b
USE_VLLM=true
GPU_MEMORY_UTILIZATION=0.9

# Supabase (pre databázu objednávok)
CORE_SUPABASE_URL=your_supabase_url
CORE_SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
TENANT_ID=your_tenant_id

# Server
PORT=8000
CORS_ALLOW_ORIGINS=*

# Twilio (voliteľné - pre produkciu)
# TWILIO_ACCOUNT_SID=your_account_sid
# TWILIO_AUTH_TOKEN=your_auth_token
```

### 3. Upravte system prompt (voliteľné)

Systémový prompt je v súbore `prompt.txt`. Môžeš ho upraviť podľa svojich potrieb.

## 🏃 Spustenie

### Lokálny development server

```bash
python main_gemma4.py
```

Server beží na `http://localhost:8000`

### Produkčné nasadenie

```bash
uvicorn main_gemma4:app --host 0.0.0.0 --port 8000 --workers 1
```

**Poznámka**: Pre GPU modely používaj iba 1 worker! Viacero workerov by spôsobilo out-of-memory chyby.

## 🔧 Architektúra

### Súborová štruktúra

```
.
├── main_gemma4.py      # FastAPI aplikácia + WebSocket handler
├── engine.py           # Gemma 4 engine (vLLM/Edge SDK wrapper)
├── utils.py            # Audio konverzie (mu-law ↔ PCM)
├── prompt.txt          # System prompt pre model
├── requirements_gemma4.txt
└── README_GEMMA4.md
```

### Tok dát

```
Twilio (mu-law 8kHz)
    ↓
WebSocket (/media-stream)
    ↓
utils.mulaw_to_pcm16() → PCM 16kHz
    ↓
engine.Gemma4Engine
    ↓
vLLM / google-ai-edge SDK
    ↓
Gemma 4 Model (GPU)
    ↓
Audio output (PCM 16kHz)
    ↓
utils.pcm16_to_mulaw() → mu-law 8kHz
    ↓
WebSocket → Twilio
```

## 🎤 Twilio integrácia

### 1. Nastav Twilio webhook

V Twilio Console nastav Voice webhook na:
```
https://your-domain.com/twilio/voice
```

### 2. Testovanie s ngrok (lokálne)

```bash
# Spusti ngrok
ngrok http 8000

# Nastav Twilio webhook na ngrok URL
https://your-ngrok-url.ngrok.io/twilio/voice
```

## 🐛 Troubleshooting

### Out of Memory (OOM) chyby

1. **Zníž GPU memory utilization**:
   ```env
   GPU_MEMORY_UTILIZATION=0.7
   ```

2. **Použi menší model**:
   ```env
   GEMMA4_MODEL_PATH=./models/gemma-4-e2b  # 2B namiesto 4B
   ```

3. **Skontroluj VRAM**:
   ```bash
   nvidia-smi
   ```

### vLLM sa nenainštaluje

Ak máš problémy s vLLM, použi google-ai-edge SDK:

```bash
pip uninstall vllm
pip install google-ai-edge
```

A nastav v `.env`:
```env
USE_VLLM=false
```

### Audio kvalita je zlá

1. **Skontroluj audio konverziu**: Uisti sa, že `audioop-lts` je nainštalovaný
2. **Zvýš threshold pre VAD**: V `main_gemma4.py` zmeň `threshold=0.02` na vyššiu hodnotu
3. **Testuj s rôznymi audio formátmi**

### Model negeneruje audio

1. **Skontroluj model path**: Uisti sa, že model je správne stiahnutý
2. **Skontroluj CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Pozri logy**: Skontroluj console output pre chybové hlášky

## 📊 Performance optimalizácia

### Zníženie latencie

1. **Použiť E2B variant** (menší, rýchlejší)
2. **Znížiť max_model_len** v `engine.py`
3. **Optimalizovať VAD threshold** pre rýchlejšiu detekciu ticha
4. **Použiť FP16/BF16** namiesto FP32

### Zvýšenie kvality

1. **Použiť E4B variant** (väčší, presnejší)
2. **Zvýšiť temperature** pre prirodzenejšie odpovede
3. **Pridať viac kontextu** do system promptu
4. **Fine-tune model** na slovenčinu (pokročilé)

## 🔐 Bezpečnosť

- **Nikdy nezdieľaj** `.env` súbor
- **Používaj HTTPS** v produkcii
- **Validuj vstupy** od používateľov
- **Limituj rate** pre API endpointy
- **Monitoruj GPU usage** pre DoS útoky

## 📝 Licencia

Tento projekt používa Gemma 4 model, ktorý má vlastnú licenciu od Google. Prečítaj si [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## 🤝 Podpora

Ak máš problémy alebo otázky:
1. Skontroluj tento README
2. Pozri logy v console
3. Otvor issue na GitHube

## 🎯 Ďalšie kroky

- [ ] Fine-tuning na slovenčinu
- [ ] Optimalizácia pre nižšiu latenciu
- [ ] Podpora pre viacero súčasných hovorov
- [ ] Monitoring a analytics
- [ ] A/B testing rôznych promptov
- [ ] Integrácia s ďalšími platformami (WhatsApp, Messenger, atď.)

---

**Poznámka**: Tento projekt je v aktívnom vývoji. Gemma 4 modely sú nové a API sa môže meniť.
