import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import tempfile
from datetime import datetime
import plotly.express as px
import pandas as pd
import json
import time
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Missing Person Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🔍 Smart Surveillance System</h1>
    <p style="color: white; margin: 0; opacity: 0.9;">
        Advanced Face Recognition for Missing Person Detection
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_models():
    """Load FaceNet and MTCNN face detector"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Use MTCNN for face detection (more reliable than YOLOv8-face for movies)
        mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
            select_largest=False  # Keep all faces
        )
        
        # Load FaceNet for recognition
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        return resnet, mtcnn, device, True
    
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, False

# ============================================================================
# FACE PROCESSING FUNCTIONS
# ============================================================================
def get_face_embedding(img: Image.Image, resnet, device):
    """Extract face embedding from PIL image"""
    try:
        # Resize and normalize
        img_resized = img.resize((160, 160))
        img_array = np.array(img_resized)
        
        # Convert to tensor
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Normalize (standard FaceNet preprocessing)
        img_tensor = (img_tensor - 127.5) / 128.0
        
        # Get embedding
        with torch.no_grad():
            embedding = resnet(img_tensor)
        
        return embedding.squeeze().cpu().numpy()
    
    except Exception as e:
        return None

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity"""
    if emb1 is None or emb2 is None:
        return 0.0
    
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot / (norm1 * norm2))

def draw_box(frame, bbox, confidence, label="Match"):
    """Draw detection box on frame"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Color by confidence
    if confidence >= 0.8:
        color = (0, 255, 0)
    elif confidence >= 0.6:
        color = (255, 165, 0)
    else:
        color = (255, 0, 0)
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw label
    text = f"{label} {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return frame

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📤 Upload Files")
    
    ref_image_file = st.file_uploader(
        "Reference Image", 
        type=["jpg", "jpeg", "png"],
        help="Clear frontal photo of missing person"
    )
    
    video_file = st.file_uploader(
        "Video File", 
        type=["mp4", "avi", "mov"],
        help="CCTV footage or video to analyze"
    )
    
    st.markdown("---")
    st.header("⚙️ Settings")
    
    sample_rate = st.slider(
        "Frame Sampling Rate", 
        min_value=1, 
        max_value=30, 
        value=5,
        help="Check every Nth frame (higher = faster but less accurate)"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.3, 
        max_value=0.9, 
        value=0.55,
        step=0.05,
        help="Minimum similarity to report a match"
    )
    
    min_face_size = st.slider(
        "Minimum Face Size (pixels)",
        min_value=20,
        max_value=100,
        value=40,
        help="Skip faces smaller than this"
    )
    
    st.markdown("---")
    st.header("📊 System Info")
    device_info = "🎮 GPU Available" if torch.cuda.is_available() else "💻 CPU Mode"
    st.info(device_info)

# ============================================================================
# MAIN PROCESSING
# ============================================================================
if ref_image_file and video_file:
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        resnet, mtcnn, device, models_loaded = load_models()
        
        if not models_loaded:
            st.error("❌ Failed to load models. Please check your environment.")
            st.stop()
    
    col1, col2 = st.columns(2)
    
    # Process reference image
    with col1:
        st.subheader("📸 Reference Image")
        ref_img = Image.open(ref_image_file).convert("RGB")
        st.image(ref_img, use_column_width=True)
        
        with st.spinner("Extracting reference face..."):
            # Detect face in reference image
            ref_boxes, ref_probs = mtcnn.detect(ref_img)
            
            if ref_boxes is None or len(ref_boxes) == 0:
                st.error("❌ No face detected in reference image!")
                st.info("💡 Tips:\n- Use a clear frontal photo\n- Ensure good lighting\n- Face should be clearly visible")
                st.stop()
            
            # Use the largest/most confident face
            best_idx = np.argmax(ref_probs) if len(ref_probs) > 1 else 0
            ref_box = ref_boxes[best_idx]
            
            # Crop face
            x1, y1, x2, y2 = map(int, ref_box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ref_img.width, x2), min(ref_img.height, y2)
            
            ref_face = ref_img.crop((x1, y1, x2, y2))
            
            # Get embedding
            ref_embedding = get_face_embedding(ref_face, resnet, device)
            
            if ref_embedding is None:
                st.error("❌ Failed to extract face features!")
                st.stop()
            
            st.success(f"✅ Reference face encoded (Confidence: {ref_probs[best_idx]*100:.1f}%)")
            
            # Show cropped face
            with st.expander("View Detected Face"):
                st.image(ref_face, caption="Extracted Face", width=200)
    
    # Video info
    with col2:
        st.subheader("🎥 Video Information")
        
        # Save video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Failed to open video file!")
            st.stop()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        st.markdown(f"""
        📊 **Properties:**
        - Duration: **{int(duration//60)}:{int(duration%60):02d}**
        - Resolution: **{width}x{height}**
        - FPS: **{fps:.1f}**
        - Total Frames: **{total_frames:,}**
        - Frames to Check: **~{total_frames//sample_rate:,}**
        """)
        
        cap.release()
    
    st.markdown("---")
    
    # Process button
    if st.button("🚀 Start Detection", type="primary", use_container_width=True):
        
        start_time = time.time()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_container = st.empty()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        processed_count = 0
        detections = []
        
        # Processing loop
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Detect faces
                boxes, probs = mtcnn.detect(frame_pil)
                
                if boxes is not None:
                    for i, (box, prob) in enumerate(zip(boxes, probs)):
                        
                        if prob < 0.9:  # Skip low-confidence face detections
                            continue
                        
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Check face size
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        if face_width < min_face_size or face_height < min_face_size:
                            continue
                        
                        # Ensure coordinates are valid
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        # Crop face
                        face_crop = frame_rgb[y1:y2, x1:x2]
                        
                        if face_crop.size == 0:
                            continue
                        
                        face_pil = Image.fromarray(face_crop)
                        
                        # Get embedding
                        face_embedding = get_face_embedding(face_pil, resnet, device)
                        
                        if face_embedding is None:
                            continue
                        
                        # Calculate similarity
                        similarity = cosine_similarity(ref_embedding, face_embedding)
                        
                        # Check threshold
                        if similarity >= confidence_threshold:
                            
                            timestamp = frame_count / fps
                            
                            # Draw box on frame
                            annotated_frame = frame_bgr.copy()
                            annotated_frame = draw_box(annotated_frame, [x1, y1, x2, y2], similarity)
                            
                            # Store detection
                            detections.append({
                                'frame_number': frame_count,
                                'timestamp': timestamp,
                                'confidence': similarity,
                                'bbox': [x1, y1, x2, y2],
                                'face_image': face_pil,
                                'annotated_frame': cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            })
                
                processed_count += 1
            
            frame_count += 1
            
            # Update progress
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            
            status_text.text(
                f"⏳ Frame {frame_count:,}/{total_frames:,} ({progress}%) | "
                f"Speed: {fps_current:.1f} FPS | Matches: {len(detections)}"
            )
            
            # Update stats every 50 frames
            if frame_count % 50 == 0 and len(detections) > 0:
                stats_container.markdown(f"""
                **Current Stats:**
                - Processed: {processed_count:,} frames
                - Detections: {len(detections)}
                - Avg Confidence: {np.mean([d['confidence'] for d in detections])*100:.1f}%
                """)
        
        cap.release()
        processing_time = time.time() - start_time
        
        status_text.text("✅ Processing Complete!")
        progress_bar.progress(100)
        
        st.markdown("---")
        
        # ====================================================================
        # RESULTS
        # ====================================================================
        
        st.header("📊 Detection Results")
        
        if detections:
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Detections", len(detections))
            
            with col2:
                avg_conf = np.mean([d['confidence'] for d in detections])
                st.metric("📈 Avg Confidence", f"{avg_conf*100:.1f}%")
            
            with col3:
                max_conf = max([d['confidence'] for d in detections])
                st.metric("🏆 Best Match", f"{max_conf*100:.1f}%")
            
            with col4:
                st.metric("⚡ Processing Time", f"{processing_time:.1f}s")
            
            # Timeline chart
            st.subheader("📈 Detection Timeline")
            
            df = pd.DataFrame([
                {
                    'Time (s)': d['timestamp'],
                    'Confidence (%)': d['confidence'] * 100,
                    'Frame': d['frame_number']
                }
                for d in detections
            ])
            
            fig = px.scatter(
                df,
                x='Time (s)',
                y='Confidence (%)',
                color='Confidence (%)',
                color_continuous_scale='RdYlGn',
                hover_data=['Frame'],
                title='Confidence Scores Over Time',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detection clips
            st.markdown("---")
            st.subheader("🎬 Top Detections")
            
            # Sort by confidence
            sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
            
            # Show top 10
            for i, det in enumerate(sorted_detections[:10]):
                
                with st.expander(
                    f"Detection #{i+1} | "
                    f"Time: {int(det['timestamp']//60)}:{int(det['timestamp']%60):02d} | "
                    f"Confidence: {det['confidence']*100:.1f}%",
                    expanded=(i < 3)
                ):
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(det['face_image'], caption="Detected Face", use_column_width=True)
                        st.metric("Frame Number", f"{det['frame_number']:,}")
                        st.metric("Confidence", f"{det['confidence']*100:.1f}%")
                    
                    with col2:
                        st.image(det['annotated_frame'], caption="Full Frame", use_column_width=True)
            
            # Export report
            st.markdown("---")
            st.subheader("📥 Export Report")
            
            report = {
                "video_file": video_file.name,
                "processing_date": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "total_frames": total_frames,
                "frames_processed": processed_count,
                "total_detections": len(detections),
                "average_confidence": float(avg_conf),
                "threshold_used": confidence_threshold,
                "detections": [
                    {
                        "frame": d['frame_number'],
                        "timestamp": f"{int(d['timestamp']//60)}:{int(d['timestamp']%60):02d}",
                        "confidence": float(d['confidence'])
                    }
                    for d in sorted_detections
                ]
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                text_report = f"""DETECTION REPORT
=================
Video: {video_file.name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Detections: {len(detections)}
Average Confidence: {avg_conf*100:.1f}%
Processing Time: {processing_time:.1f}s

TOP 10 MATCHES
--------------
"""
                for i, d in enumerate(sorted_detections[:10]):
                    text_report += f"{i+1}. Frame {d['frame_number']:,} | Time: {int(d['timestamp']//60)}:{int(d['timestamp']%60):02d} | Confidence: {d['confidence']*100:.1f}%\n"
                
                st.download_button(
                    label="📋 Download Text Report",
                    data=text_report,
                    file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        else:
            st.warning("⚠️ No matches found!")
            st.markdown("""
            **Troubleshooting Tips:**
            - ✅ Lower the confidence threshold (try 0.4-0.5)
            - ✅ Check if faces are clearly visible in video
            - ✅ Ensure reference image is a clear frontal photo
            - ✅ Try reducing frame sampling rate (check more frames)
            - ✅ Verify video quality is good enough
            """)

else:
    st.info("👈 Upload a reference image and video file from the sidebar to begin analysis")
    
    st.markdown("### 🎯 System Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Advanced Detection**
        - MTCNN face detection
        - FaceNet recognition (VGGFace2)
        - 99%+ accuracy capability
        """)
    
    with col2:
        st.markdown("""
        **⚡ Smart Processing**
        - Configurable frame sampling
        - Real-time progress tracking
        - GPU acceleration support
        """)
    
    with col3:
        st.markdown("""
        **📊 Professional Output**
        - Interactive timeline charts
        - Confidence analytics
        - Exportable JSON/Text reports
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>Smart Surveillance System</strong> | Powered by FaceNet & MTCNN</p>
</div>
""", unsafe_allow_html=True)