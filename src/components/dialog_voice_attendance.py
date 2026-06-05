import streamlit as st

from src.pipelines.voice_pipeline import (
    process_bulk_audio,
    classify_voice_presence,
    VOICE_MATCH_THRESHOLD,
    VOICE_RELATIVE_MARGIN,
    has_valid_voice_profile,
)

from src.database.config import supabase

import pandas as pd


from src.components.dialog_attendance_results import show_attendance_result
from datetime import datetime
@st.dialog('Voice Attendance')
def voice_attendance_dialog(selected_subject_id):
    st.write('Record audio of students saying I am present. Then AI will recognize the students')


    audio_data = None

    audio_data = st.audio_input("Record classroom audio")

    if st.button('Analyze Audio', width='stretch', type='primary'):
        with st.spinner('Prcessing Audio data'):
            enrolled_res = supabase.table('subject_students').select("*, students(*)").eq('subject_id',selected_subject_id ).execute()
            enrolled_students = enrolled_res.data

            if not enrolled_students:
                st.warning('No students enrolled in this course')
                return
            candidates_dict = {
                int(s['students']['student_id']): s['students']['voice_embedding']
                for s in enrolled_students
                if s['students'].get('voice_embedding')
            }

            missing_voice = [
                s['students']['name']
                for s in enrolled_students
                if not s['students'].get('voice_embedding')
            ]
            invalid_voice = [
                s['students']['name']
                for s in enrolled_students
                if s['students'].get('voice_embedding')
                and not has_valid_voice_profile(s['students']['voice_embedding'])
            ]
            if missing_voice:
                st.warning(
                    "No voice profile for: "
                    + ", ".join(missing_voice)
                    + ". Re-register with voice enrollment."
                )
            if invalid_voice:
                st.warning(
                    "Invalid voice profile (re-record): " + ", ".join(invalid_voice)
                )

            if not candidates_dict:
                st.error('No enrolled students have voice profiles registerd')
                return
            
            audio_bytes = audio_data.read()

            student_scores = process_bulk_audio(audio_bytes, candidates_dict)
            presence = classify_voice_presence(student_scores)

            results, attendance_to_log = [], []

            current_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            max_score = max(student_scores.values()) if student_scores else 0.0
            cutoff = max(VOICE_MATCH_THRESHOLD, max_score - VOICE_RELATIVE_MARGIN)

            for node in enrolled_students:
                student = node['students']
                sid = int(student['student_id'])
                score = float(student_scores.get(sid, 0.0))
                is_present = presence.get(sid, False)

                if is_present:
                    source = f"{score:.3f}"
                elif score < VOICE_MATCH_THRESHOLD:
                    source = f"{score:.3f} (need ≥{VOICE_MATCH_THRESHOLD})"
                else:
                    source = f"{score:.3f} (need ≥{cutoff:.3f}, top {max_score:.3f})"

                results.append({
                    "Name": student['name'],
                    "ID": student['student_id'],
                    "Source": source,
                    "Status": "✅ Present" if is_present else "❌ Absent",
                })

                attendance_to_log.append({
                    'student_id': student['student_id'],
                    'subject_id': selected_subject_id,
                    'timestamp': current_timestamp,
                    'is_present': bool(is_present)
                })
            st.session_state.voice_attendance_results = (pd.DataFrame(results), attendance_to_log)

    if st.session_state.get('voice_attendance_results'):
        st.divider()
        df_results, logs = st.session_state.voice_attendance_results
        show_attendance_result(df_results, logs)

