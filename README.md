# Voice_Cloning_Script
````markdown
# Voice_Cloning_Script
# 🎙️ Voice Cloning with Open-Source TTS Models

Clone any voice using just a few seconds of reference audio with state-of-the-art open-source models like **XTTS-v2** and **Kokoro-82M**. This project supports real-time voice synthesis, quality evaluation, dataset preprocessing, benchmarking, and an interactive **Gradio UI**.

## 🚀 Features

- ✅ Voice Cloning with XTTS-v2
- ✅ Reference audio support
- ✅ Multilingual synthesis
- ✅ Real-Time Factor (RTF) benchmarking
- ✅ Mel-Cepstral Distortion (MCD) & Prosody evaluation
- ✅ Dataset preprocessing + metadata creation
- ✅ Fine-tuning ready (train config builder included)
- ✅ Gradio interface for testing in your browser

---

## 📦 Tech Stack

- 🧠 [Coqui TTS](https://github.com/coqui-ai/TTS)
- 🤖 HuggingFace Transformers
- 🎵 Librosa, Soundfile
- 📊 Parselmouth for prosody analysis
- 🛠️ PyTorch, ONNX, Accelerate
- 🌐 Gradio for UI

---

## 🛠️ Setup Instructions

### 🔧 Install Dependencies

```bash
pip install -q transformers accelerate datasets torch torchvision torchaudio
pip install -q TTS coqui-tts librosa soundfile phonemizer espeak-ng
pip install -q wandb parselmouth praat-parselmouth
pip install -q huggingface_hub safetensors onnx onnxruntime-gpu gradio ipywidgets
apt-get -qq -y install espeak-ng
````

---

## 🧪 How It Works

1. **Load Models**
   Loads XTTS-v2 and Kokoro-82M with GPU support.

2. **Process Dataset**
   Downloads sample dataset (LibriSpeech), trims silence, normalizes, and splits into 5–10s chunks.

3. **Generate Reference Audio**
   Uses default XTTS voice to synthesize a reference clip.

4. **Clone the Voice**
   Takes custom text + reference audio and generates cloned speech.

5. **Evaluate the Output**
   Calculates MCD and pitch similarity. Benchmarks inference time & RTF.

---

## 🎯 Voice Cloning Demo

```python
text = "Hello, this is a voice cloning demo."
reference_audio = "reference.wav"
output_path = "cloned_output.wav"

result = cloner.clone_voice_xtts(text, reference_audio, output_path, language="en")
```

---

## 🎛️ Gradio Interface

Launch it in your browser:

```bash
python voice_cloning_script.py
```

Then upload a `.wav` file and type the text you want spoken!

---

## 📊 Evaluation Metrics

| Metric                | Description                      |
| --------------------- | -------------------------------- |
| ⏱️ Inference Time     | How long it takes to synthesize  |
| 🔁 Real-Time Factor   | Inference time / audio duration  |
| 🎼 Mel-Cepstral Dist. | Similarity in spectral features  |
| 📈 Pitch Similarity   | Matches speaker tone & frequency |

---

## 🤝 Contributing

Want to contribute or fine-tune this further? Pull requests are welcome!


## ✨ Author

**Hafiza Aliza Mustafa**
📧 [aleezamustafa11@gmail.com](mailto:aleezamustafa11@gmail.com)
🌍 Karachi, Pakistan

```

