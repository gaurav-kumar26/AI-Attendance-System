import io

import streamlit as st


def _storage_key(key):
    return f"{key}_photo"


def _to_upload_file(data):
    buf = io.BytesIO(data)
    buf.name = "photo.jpg"
    buf.seek(0)
    return buf


def clear_face_photo(key):
    storage_key = _storage_key(key)
    st.session_state.pop(storage_key, None)
    st.session_state.pop(f"{key}_processed", None)
    st.session_state.pop(f"{key}_register", None)


def get_stored_face_photo(key):
    storage_key = _storage_key(key)
    if storage_key in st.session_state:
        return _to_upload_file(st.session_state[storage_key])
    return None


def face_photo_input(label="Position your face in the center", key="face_photo", show_camera=True):
    """Browser webcam capture matching the FaceID login UI."""
    storage_key = _storage_key(key)
    stored = get_stored_face_photo(key)

    if stored and not show_camera:
        stored.seek(0)
        st.image(stored, use_container_width=True)
        if st.button("Retake Photo", key=f"{key}_retake", type="secondary"):
            clear_face_photo(key)
            st.rerun()
        stored.seek(0)
        return stored

    if stored:
        stored.seek(0)
        return stored

    st.markdown(
        f'<p style="text-align:center;font-size:1.05rem;color:#4a4a4a;margin-bottom:0.5rem;">{label}</p>',
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 3, 1])
    with center:
        camera_photo = st.camera_input(
            label,
            key=f"{key}_browser",
            label_visibility="collapsed",
        )
        if camera_photo is not None:
            st.session_state[storage_key] = camera_photo.getvalue()
            st.session_state.pop(f"{key}_processed", None)
            st.rerun()

    return None
