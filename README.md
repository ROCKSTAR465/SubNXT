# 🎬 AI Subtitle Generator — SubNXT

**SubNXT** is an AI-powered web application that allows users to upload videos or audio files and generate accurate, editable subtitles using OpenAI's Whisper model. Built with a modern UI in Streamlit, SubNXT helps content creators, educators, and professionals save time by automating subtitle generation and export.

---

## 🌟 Features

✅ **Supports multiple media formats**: `mp4`, `mp3`, `avi`, `mov`, `mkv`, `wav`, etc.  
✅ **Automatic transcription + translation** using Whisper's multilingual ASR  
✅ **Select model quality**: Choose from Tiny, Base, Small, Medium, or Large Whisper models  
✅ **Real-time progress display** during subtitle generation  
✅ **Download subtitles** as `.vtt` (WebVTT format) or `.json`  
✅ **Edit subtitles manually** in an intuitive expandable panel  
✅ **Play video inline** with generated subtitles  
✅ **Responsive and modern UI** using custom Streamlit CSS  
✅ **Temp file management**: Uploads are stored temporarily and securely  

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/) – Interactive web interface  
- [OpenAI Whisper](https://github.com/openai/whisper) – Speech recognition and translation  
- [moviepy](https://zulko.github.io/moviepy/) – Video file handling  
- `tempfile`, `json`, `os`, and `time` – Standard Python libraries  

---

## 🚀 How to Run Locally

### Prerequisites

Make sure you have Python 3.10 or higher installed.

```bash
pip install -r requirements.txt

### **1. Run the Web App**
Launch the Streamlit app locally:
```bash
streamlit run SubNXT.py
```

Once the app is running, open your browser and go to:
```
http://localhost:8501
```
---

## **Folder Structure**

SubNXT/
├── SubNXT.py             # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and usage
└── .streamlit/            # (Optional) Streamlit config directory

