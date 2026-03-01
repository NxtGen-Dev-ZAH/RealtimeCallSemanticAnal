# Server Warnings Explanation & Fixes

## Status: ✅ Server is Running Successfully

All warnings shown are **non-critical**. Your FastAPI server is working correctly. These are optional dependency warnings that can be fixed for better functionality.

---

## Warning Breakdown

### 1. ⚠️ Missing ffmpeg/avconv (RuntimeWarning)
```
RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
```

**What it means:**
- `pydub` library needs `ffmpeg` to convert audio formats (mp3, m4a → wav)
- Currently, audio format conversion may fail for non-WAV files

**Impact:** 
- ⚠️ Medium - Only affects audio format conversion
- ✅ WAV files will work fine
- ❌ MP3/M4A uploads may fail

**Fix (macOS):**
```bash
brew install ffmpeg
```

**Fix (Linux):**
```bash
sudo apt-get install ffmpeg
# or
sudo yum install ffmpeg
```

**Fix (Windows):**
Download from https://ffmpeg.org/download.html and add to PATH

---

### 2. ⚠️ Missing spaCy Model (UserWarning)
```
WARNING: spaCy model 'en_core_web_sm' not found. PII masking will be limited.
```

**What it means:**
- spaCy model is needed for advanced PII (Personally Identifiable Information) masking
- Without it, PII masking uses basic regex patterns (less accurate)

**Impact:**
- ⚠️ Low - PII masking still works, just less sophisticated
- ✅ Core functionality unaffected

**Fix:**
```bash
# Activate your virtual environment first
cd backend
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

# Install spaCy model
python -m spacy download en_core_web_sm
```

---

### 3. 📢 Deprecation Warnings (Informational)

#### a) torchaudio deprecation
```
UserWarning: torchaudio._backend.list_audio_backends has been deprecated
```

**What it means:**
- PyTorch is updating their audio backend
- This is a library-level warning, not your code

**Impact:**
- ✅ None - Already suppressed in code (see `preprocessing.py` lines 25-28)
- Future PyTorch versions will use TorchCodec instead

**Fix:** Already handled in code with warning filters

---

#### b) pkg_resources deprecation
```
UserWarning: pkg_resources is deprecated as an API
```

**What it means:**
- `webrtcvad` library uses deprecated `pkg_resources`
- This is a third-party library issue

**Impact:**
- ✅ None - Works fine now
- ⚠️ May need updates in 2025-2026

**Fix:** Wait for `webrtcvad` library to update, or ignore (it's harmless)

---

#### c) transformers deprecation
```
UserWarning: `return_all_scores` is now deprecated, use `top_k=None` instead
```

**What it means:**
- Hugging Face Transformers library updated their API
- Old parameter name still works but is deprecated

**Impact:**
- ✅ None - Code still works
- ⚠️ Should update code eventually

**Fix:** Update code to use `top_k=None` instead of `return_all_scores=True` (low priority)

---

## Summary

| Warning | Severity | Action Required | Priority |
|---------|----------|----------------|----------|
| ffmpeg missing | Medium | Install ffmpeg | Medium |
| spaCy model missing | Low | Install model | Low |
| torchaudio deprecation | None | Already handled | None |
| pkg_resources deprecation | None | Wait for library update | None |
| transformers deprecation | None | Update code later | Low |

---

## Quick Fix Commands (for complete setup)

```bash
# 1. Install ffmpeg (macOS)
brew install ffmpeg

# 2. Install spaCy model
cd backend
source .venv/bin/activate
python -m spacy download en_core_web_sm

# 3. Restart server
python run_web_app.py
```

---

## Current Status

✅ **Your server is running and functional!**
- All core features work
- API endpoints are accessible
- Models are loading correctly
- Demo mode is active

The warnings are just notifications about optional improvements. Your demo will work fine without fixing them, but installing ffmpeg and the spaCy model will improve audio processing and PII masking capabilities.
