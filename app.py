import io
import os
import json
import tempfile
from datetime import datetime, timedelta
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
from docx import Document
from fpdf import FPDF

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Labour Court ASR & Translate",
    page_icon="⚖️",
    layout="wide"
)

ARCHIVE_DIR = "court_archive"
if not os.path.exists(ARCHIVE_DIR):
    os.makedirs(ARCHIVE_DIR)

# Cache engines to avoid reloading on every Streamlit rerun
@st.cache_resource
def load_whisper_model(model_size="base"):
    return WhisperModel(model_size, device="cpu", compute_type="int8")

@st.cache_resource
def load_translation_components(source_code, target_code):
    model_name = f"Helsinki-NLP/opus-mt-{source_code}-{target_code}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

model = load_whisper_model()

# -----------------------------------------------------------------------------
# Export Generator Functions & Helper UI Component
# -----------------------------------------------------------------------------
def create_docx_bytes(content_text: str, suit_number: str) -> bytes:
    """Generates a structured Word document in memory."""
    doc = Document()
    doc.add_heading("Ghana Labour Court Official Transcript", level=1)
    
    p_meta = doc.add_paragraph()
    p_meta.add_run("Suit Number: ").bold = True
    p_meta.add_run(f"{suit_number}\n")
    p_meta.add_run("Exported On: ").bold = True
    p_meta.add_run(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    doc.add_heading("Transcript Content", level=2)
    
    for line in content_text.split("\n"):
        if line.strip():
            doc.add_paragraph(line)
            
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def create_pdf_bytes(content_text: str, suit_number: str) -> bytes:
    """Generates a formatted PDF document in memory using fpdf2."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(0, 10, "Ghana Labour Court Official Transcript", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, f"Suit Number: {suit_number}\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.ln(5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=10)
    for line in content_text.split("\n"):
        if line.strip():
            clean_line = line.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 6, clean_line)
            pdf.ln(1)
            
    return bytes(pdf.output())


def render_export_dropdown(content_text: str, suit_number: str, key_suffix: str):
    """Renders a dropdown list to choose export format and a dynamic export button."""
    col_fmt, col_btn = st.columns([1, 1])
    
    with col_fmt:
        export_fmt = st.selectbox(
            "Select Format:",
            options=["Text (.txt)", "Word (.docx)", "PDF (.pdf)"],
            key=f"fmt_select_{key_suffix}"
        )
        
    suit_clean = suit_number.replace('/', '_').replace(' ', '_')
    
    with col_btn:
        st.write("") # Alignment spacing
        st.write("")
        if export_fmt == "Text (.txt)":
            st.download_button(
                label="📤 Export .txt",
                data=content_text,
                file_name=f"{suit_clean}.txt",
                mime="text/plain",
                key=f"dl_txt_{key_suffix}"
            )
        elif export_fmt == "Word (.docx)":
            docx_data = create_docx_bytes(content_text, suit_number)
            st.download_button(
                label="📄 Export .docx",
                data=docx_data,
                file_name=f"{suit_clean}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_docx_{key_suffix}"
            )
        elif export_fmt == "PDF (.pdf)":
            pdf_data = create_pdf_bytes(content_text, suit_number)
            st.download_button(
                label="📕 Export .pdf",
                data=pdf_data,
                file_name=f"{suit_clean}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{key_suffix}"
            )

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Transcription Settings")
    translate_option = st.checkbox(
        "Whisper Real-Time English Translate",
        value=False,
        help="Translates non-English audio straight into English text during ASR."
    )

# -----------------------------------------------------------------------------
# Core Helper Functions
# -----------------------------------------------------------------------------
def format_official_banner(metadata):
    return (
        "==================================================\n"
        "GHANA LABOUR COURT OFFICIAL TRANSCRIPT\n"
        f"Suit Number: {metadata['suit_number']}\n"
        f"Presiding Judge: {metadata['judge_name']}\n"
        f"Proceeding Type: {metadata['proceeding_type']}\n"
        f"Session Date: {metadata['session_date']}\n"
        f"Recorder Name: {metadata['recorded_by']}\n"
        "==================================================\n\n"
    )

def format_timestamp(seconds):
    """Converts seconds (float) into HH:MM:SS format."""
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def transcribe_audio_with_faster_whisper(audio_path, translate=False):
    task_type = "translate" if translate else "transcribe"
    segments, info = model.transcribe(audio_path, task=task_type, beam_size=5)
    
    formatted_segments = []
    for segment in segments:
        start_str = format_timestamp(segment.start)
        end_str = format_timestamp(segment.end)
        line = f"[{start_str} -> {end_str}] {segment.text.strip()}"
        formatted_segments.append(line)
        
    detected_lang = info.language.upper()
    confidence = info.language_probability * 100
    
    return "\n".join(formatted_segments), detected_lang, confidence

def save_session_to_archive(metadata, raw_transcript, detected_lang="N/A", confidence=0.0):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    clean_suit = metadata["suit_number"].replace("/", "-").replace(" ", "_")
    
    base_filename = f"{timestamp_str}_{clean_suit}"
    filepath = os.path.join(ARCHIVE_DIR, f"{base_filename}.json")
    
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(ARCHIVE_DIR, f"{base_filename}_v{counter}.json")
        counter += 1

    banner = format_official_banner(metadata)
    full_transcript_text = banner + raw_transcript

    record = {
        "filename": os.path.basename(filepath),
        "metadata": metadata,
        "detected_language": detected_lang,
        "language_confidence": f"{confidence:.1f}%",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transcript": full_transcript_text
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=4)
        
    st.cache_data.clear()
    return filepath, full_transcript_text

@st.cache_data
def load_archive_records():
    records = []
    if os.path.exists(ARCHIVE_DIR):
        for filename in os.listdir(ARCHIVE_DIR):
            if filename.endswith(".json"):
                path = os.path.join(ARCHIVE_DIR, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["filename"] = filename
                        records.append(data)
                except Exception:
                    pass
    return sorted(records, key=lambda x: x.get("created_at", ""), reverse=True)

def render_metadata_inputs(key_prefix):
    st.subheader("📋 Session Metadata")
    col1, col2 = st.columns(2)
    with col1:
        suit_number = st.text_input("Suit Number", value="SUIT NO. LC/001/2026", key=f"suit_{key_prefix}")
        judge_name = st.text_input("Presiding Judge", value="Justice A. Mensah", key=f"judge_{key_prefix}")
        recorded_by = st.text_input("Recorder Name", value="Official Clerk", key=f"clerk_{key_prefix}")
    with col2:
        proceeding_type = st.selectbox(
            "Proceeding Type",
            ["Witness Testimony", "Cross-Examination", "Bail Application", "Ruling / Judgment", "Motion Argument"],
            key=f"proc_{key_prefix}"
        )
        session_date = st.date_input("Session Date", key=f"date_{key_prefix}")
        
    return {
        "suit_number": suit_number,
        "judge_name": judge_name,
        "proceeding_type": proceeding_type,
        "session_date": str(session_date),
        "recorded_by": recorded_by
    }

# -----------------------------------------------------------------------------
# Standalone Translation Studio Module
# -----------------------------------------------------------------------------
def handle_translation_ui(full_transcript_text, key_suffix):
    st.markdown("---")
    st.subheader("🌐 Offline Translation Studio")
    st.caption("Translate speech transcript body (metadata heading is automatically stripped).")
    
    lines = full_transcript_text.split("\n")
    cleaned_lines = [
        line for line in lines 
        if not any(header_key in line for header_key in [
            "=======", "GHANA LABOUR COURT", "Suit Number:", 
            "Presiding Judge:", "Proceeding Type:", "Session Date:", "Recorder Name:"
        ])
    ]
    transcript_body = "\n".join(cleaned_lines).strip()
    
    lang_map = {
        "English": "en", "French": "fr", "German": "de", "Spanish": "es", 
        "Arabic": "ar", "Chinese (Simplified)": "zh", "Dutch": "nl", 
        "Italian": "it", "Japanese": "ja", "Portuguese": "pt", "Russian": "ru",
        "Afrikaans": "af", "Bulgarian": "bg", "Croatian": "hr", "Czech": "cs", 
        "Danish": "da", "Estonian": "et", "Finnish": "fi", "Greek": "el", 
        "Hausa 🇳🇬/🇬🇭": "ha", "Hindi": "hi", "Hungarian": "hu", "Indonesian": "id", 
        "Korean": "ko", "Lithuanian": "lt", "Norwegian": "no", "Polish": "pl", 
        "Romanian": "ro", "Slovak": "sk", "Slovenian": "sl", "Swedish": "sv", 
        "Turkish": "tr", "Ukrainian": "uk"
    }
    
    sorted_languages = sorted(list(lang_map.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox("From (Source):", sorted_languages, index=sorted_languages.index("English"), key=f"src_{key_suffix}")
    with col2:
        tgt_lang = st.selectbox("To (Target):", sorted_languages, index=sorted_languages.index("French"), key=f"tgt_{key_suffix}")
        
    from_code = lang_map[src_lang]
    to_code = lang_map[tgt_lang]
    
    if st.button(f"Run Translation ({src_lang} ➔ {tgt_lang})", key=f"btn_{key_suffix}", type="secondary"):
        if from_code == to_code:
            st.warning("Source and Target languages are identical. No translation needed!")
        else:
            try:
                with st.spinner(f"Loading local {src_lang} to {tgt_lang} translation model..."):
                    tokenizer, model_components = load_translation_components(from_code, to_code)
                    
                with st.spinner("Translating speech transcript line-by-line..."):
                    body_lines = transcript_body.split("\n")
                    translated_lines = []
                    
                    for line in body_lines:
                        if line.strip():
                            timestamp, text_content = "", line
                            if "]" in line:
                                timestamp, text_content = line.split("]", 1)
                                timestamp += "]"
                            
                            if text_content.strip():
                                inputs = tokenizer(text_content.strip(), return_tensors="pt")
                                with torch.no_grad():
                                    generated_tokens = model_components.generate(**inputs)
                                translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                                
                                translated_lines.append(f"{timestamp} {translated_text}" if timestamp else translated_text)
                            else:
                                translated_lines.append(line)
                    
                    translated_output = "\n".join(translated_lines)
                    st.success("Translation Complete!")
                    
                    compare_col1, compare_col2 = st.columns(2)
                    with compare_col1:
                        edited_orig = st.text_area(f"📝 Original Transcript Body ({src_lang})", value=transcript_body, height=250, key=f"txt_orig_comp_{key_suffix}")
                        render_export_dropdown(edited_orig, "ORIGINAL", f"trans_orig_{key_suffix}")
                        
                    with compare_col2:
                        edited_trans = st.text_area(f"📝 Translated Speech Body ({tgt_lang})", value=translated_output, height=250, key=f"txt_res_{key_suffix}")
                        render_export_dropdown(edited_trans, f"{from_code}_TO_{to_code}", f"trans_tgt_{key_suffix}")

            except Exception as e:
                st.error("⚠️ Direct Offline Translation Route Unavailable")
                st.info(
                    f"**Why this happens:** Direct paths between non-English languages (like {src_lang} to {tgt_lang}) route through English.\n\n"
                    f"**Two-step workaround:**\n"
                    f"1. Set target language to **English** and translate.\n"
                    f"2. Copy the result, set source to **English**, target to **{tgt_lang}**, and translate!"
                )

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
st.title("⚖️ Ghana Labour Court - Speech to Text & Translation")
st.caption("Run secure, local courtroom transcription and multilingual translations.")

tab1, tab2, tab3 = st.tabs([
    "📁 Upload File",
    "🎙️ Live Recording Session",
    "🗄️ Case File Archive"
])

# -----------------------------------------------------------------------------
# TAB 1: File Upload
# -----------------------------------------------------------------------------
with tab1:
    meta_upload = render_metadata_inputs(key_prefix="upload")
    
    audio_file = st.file_uploader("Choose a court audio file", type=["wav", "mp3", "m4a", "flac"], key="uploader_tab1")
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("Generate Transcript from File", type="primary", key="btn_transcribe_upload"):
            with st.spinner("Processing audio with faster-whisper..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_file.getbuffer())
                    tmp_path = tmp_file.name

                raw_text, lang, conf = transcribe_audio_with_faster_whisper(tmp_path, translate=translate_option)
                saved_path, complete_text = save_session_to_archive(meta_upload, raw_text, lang, conf)
                
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
                st.session_state['upload_complete_text'] = complete_text
                st.session_state['upload_saved_path'] = saved_path
                st.session_state['upload_lang'] = lang
                st.session_state['upload_conf'] = conf
                st.rerun()

    if 'upload_complete_text' in st.session_state:
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.success(f"Archived to `{os.path.basename(st.session_state['upload_saved_path'])}`!")
        with status_col2:
            st.info(f"🌐 **Language:** `{st.session_state['upload_lang']}` | **Confidence:** `{st.session_state['upload_conf']:.1f}%`")
        
        text_key = f"txt_area_upload_{hash(st.session_state['upload_complete_text'])}"
        st.text_area("Generated Official Transcript", value=st.session_state['upload_complete_text'], height=300, key=text_key)
        
        render_export_dropdown(st.session_state['upload_complete_text'], meta_upload['suit_number'], "tab1_main")
        
        handle_translation_ui(st.session_state['upload_complete_text'], "upload")

# -----------------------------------------------------------------------------
# TAB 2: Audio Recording
# -----------------------------------------------------------------------------
with tab2:
    meta_live = render_metadata_inputs(key_prefix="live")
    
    live_audio = st.audio_input("Record court session directly", key="audio_input_tab2")
    
    if live_audio is not None:
        st.audio(live_audio)
        
        if st.button("Generate Transcript from Live Recording", type="primary", key="btn_transcribe_live"):
            with st.spinner("Processing live session recording..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(live_audio.getbuffer())
                    tmp_path = tmp_file.name

                raw_text, lang, conf = transcribe_audio_with_faster_whisper(tmp_path, translate=translate_option)
                saved_path, complete_text = save_session_to_archive(meta_live, raw_text, lang, conf)
                
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
                st.session_state['live_complete_text'] = complete_text
                st.session_state['live_saved_path'] = saved_path
                st.session_state['live_lang'] = lang
                st.session_state['live_conf'] = conf
                st.rerun()

    if 'live_complete_text' in st.session_state:
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.success(f"Archived to `{os.path.basename(st.session_state['live_saved_path'])}`!")
        with status_col2:
            st.info(f"🌐 **Language:** `{st.session_state['live_lang']}` | **Confidence:** `{st.session_state['live_conf']:.1f}%`")
        
        text_key_live = f"txt_area_live_{hash(st.session_state['live_complete_text'])}"
        st.text_area("Generated Official Transcript", value=st.session_state['live_complete_text'], height=300, key=text_key_live)
        
        render_export_dropdown(st.session_state['live_complete_text'], meta_live['suit_number'], "tab2_main")
        
        handle_translation_ui(st.session_state['live_complete_text'], "live")

# -----------------------------------------------------------------------------
# TAB 3: Registry & Archive
# -----------------------------------------------------------------------------
with tab3:
    st.header("🗄️ Case File Archive Registry")
    records = load_archive_records()
    
    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        search_query = st.text_input("🔍 Search Suit Number or Judge Name", value="", key="arch_search_input")
    with col_filter2:
        proc_filter = st.selectbox(
            "Filter by Proceeding Type",
            ["All Types", "Witness Testimony", "Cross-Examination", "Bail Application", "Ruling / Judgment", "Motion Argument"],
            key="arch_proc_select"
        )
        
    filtered_records = []
    for rec in records:
        m = rec.get("metadata", {})
        suit = m.get("suit_number", "").lower()
        judge = m.get("judge_name", "").lower()
        p_type = m.get("proceeding_type", "")
        
        matches_query = search_query.lower() in suit or search_query.lower() in judge
        matches_type = (proc_filter == "All Types") or (proc_filter == p_type)
        
        if matches_query and matches_type:
            filtered_records.append(rec)

    st.write(f"Showing **{len(filtered_records)}** record(s)")
    st.divider()

    if not filtered_records:
        st.info("No matching records found in local archive directory.")
    else:
        for idx, rec in enumerate(filtered_records):
            meta = rec.get("metadata", {})
            created_at = rec.get('created_at', 'N/A')
            filename = rec.get('filename', '')
            suit_num = meta.get('suit_number', 'N/A')
            
            title = f"📌 {suit_num} | {meta.get('proceeding_type', 'N/A')} ({meta.get('session_date', 'N/A')}) — Archived at {created_at}"
            
            with st.expander(title):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Presiding Judge:** {meta.get('judge_name', 'N/A')}")
                    st.write(f"**Recorded By:** {meta.get('recorded_by', 'N/A')}")
                with c2:
                    st.write(f"**Detected Lang:** `{rec.get('detected_language', 'N/A')}`")
                    st.write(f"**Confidence:** `{rec.get('language_confidence', 'N/A')}`")
                
                transcript_content = rec.get("transcript", "")
                st.text_area("Full Transcript Content", value=transcript_content, height=220, key=f"preview_{filename}")
                
                exp_col1, del_col = st.columns([3, 1])
                
                with exp_col1:
                    render_export_dropdown(transcript_content, suit_num, f"arch_{filename}")
                with del_col:
                    st.write("") # Spacing
                    st.write("")
                    if st.button("🗑️ Delete", key=f"del_arch_{filename}", type="primary"):
                        target_file = os.path.join(ARCHIVE_DIR, filename)
                        if os.path.exists(target_file):
                            os.remove(target_file)
                            st.cache_data.clear()
                            st.success(f"Deleted `{filename}` successfully!")
                            st.rerun()