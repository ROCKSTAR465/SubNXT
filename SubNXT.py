import streamlit as st
import whisper
import os
import tempfile
from moviepy import VideoFileClip
import time
import json
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
# Configure Streamlit page
st.set_page_config(
    page_title="AI Subtitle Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle-text {
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-size: 1.2rem;
        text-align: center;
        margin: 10px 0;
    }
    
    .progress-container {
        margin: 20px 0;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'subtitles' not in st.session_state:
    st.session_state.subtitles = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def format_timestamp(seconds):
    """Convert seconds to VTT timestamp format"""
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def generate_subtitles(video_path, model_type):
    """Generate subtitles using Whisper"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading Whisper model...")
        progress_bar.progress(20)
        model = whisper.load_model(model_type)
        
        status_text.text("Transcribing video...")
        progress_bar.progress(40)
        result = model.transcribe(video_path, task="translate")
        
        status_text.text("Processing subtitles...")
        progress_bar.progress(80)
        
        # Format subtitles for display
        subtitles = []
        for segment in result["segments"]:
            subtitles.append({
                'start': segment["start"],
                'end': segment["end"],
                'text': segment["text"].strip()
            })
        
        progress_bar.progress(100)
        status_text.text("✅ Subtitles generated successfully!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return subtitles
        
    except Exception as e:
        st.error(f"Error generating subtitles: {str(e)}")
        return None

def get_subtitle_at_time(subtitles, current_time):
    """Get the subtitle text for the current time"""
    if not subtitles:
        return ""
    
    for subtitle in subtitles:
        if subtitle['start'] <= current_time <= subtitle['end']:
            return subtitle['text']
    return ""

def create_vtt_file(subtitles):
    """Create a VTT file content from subtitles"""
    vtt_content = "WEBVTT\n\n"
    for subtitle in subtitles:
        start_time = format_timestamp(subtitle['start'])
        end_time = format_timestamp(subtitle['end'])
        text = subtitle['text']
        vtt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    return vtt_content

# Main UI
st.markdown('<h1 class="main-header">🎬 AI Subtitle Generator</h1>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("🛠️ Configuration")
    
    model_type = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower"
    )
    
    st.info("""
    **Model Comparison:**
    - **Tiny**: Fastest, basic accuracy
    - **Base**: Good balance (recommended)
    - **Small**: Better accuracy
    - **Medium**: High accuracy
    - **Large**: Best accuracy, slowest
    """)

# File upload
uploaded_file = st.file_uploader(
    "Choose a video or audio file",
    type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v', 'mp3', 'wav', 'flac'],
    help="Upload your video or audio file to generate subtitles"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
        st.session_state.video_path = video_path
    
    # Display file info
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.success(f"📁 File uploaded: {uploaded_file.name} ({file_size:.2f} MB)")
    
    # Generate subtitles button
    if st.button("🚀 Generate Subtitles", type="primary", use_container_width=True):
        st.session_state.processing = True
        
        with st.spinner("Processing..."):
            subtitles = generate_subtitles(video_path, model_type)
            if subtitles:
                st.session_state.subtitles = subtitles
                st.session_state.processing = False
                st.rerun()

# Display video with subtitles if available
if st.session_state.video_path and st.session_state.subtitles:
    st.header("🎥 Video with Generated Subtitles")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display video
        try:
            video_file = open(st.session_state.video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()
        except Exception as e:
            st.error(f"Error displaying video: {str(e)}")
    
    with col2:
        st.subheader("📝 Subtitle Controls")
        
        # Download VTT file
        vtt_content = create_vtt_file(st.session_state.subtitles)
        st.download_button(
            label="⬇️ Download VTT File",
            data=vtt_content,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.vtt",
            mime="text/vtt",
            use_container_width=True
        )
        
        # Download JSON file
        json_content = json.dumps(st.session_state.subtitles, indent=2)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_content,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitles.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Display subtitles timeline
    st.subheader("📋 Generated Subtitles")
    
    # Create a scrollable container for subtitles
    subtitle_container = st.container()
    with subtitle_container:
        for i, subtitle in enumerate(st.session_state.subtitles):
            start_time = subtitle['start']
            end_time = subtitle['end']
            text = subtitle['text']
            
            # Format time display
            start_formatted = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
            end_formatted = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
            
            with st.expander(f"🕐 {start_formatted} - {end_formatted}", expanded=False):
                st.write(text)
                
                # Edit subtitle option
                edited_text = st.text_area(
                    "Edit subtitle:",
                    value=text,
                    key=f"edit_{i}",
                    height=70
                )
                
                if st.button(f"Update", key=f"update_{i}"):
                    st.session_state.subtitles[i]['text'] = edited_text
                    st.success("Subtitle updated!")
                    st.rerun()

elif st.session_state.video_path and not st.session_state.subtitles and not st.session_state.processing:
    st.info("👆 Click 'Generate Subtitles' to create subtitles for your video!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit and OpenAI Whisper"
    "</div>",
    unsafe_allow_html=True
)
