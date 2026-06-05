

import dlib
import numpy as np
import face_recognition_models
import streamlit as st

from src.database.db import get_all_students

# Login: strict — avoid wrong account
LOGIN_THRESHOLD = 0.45
LOGIN_MIN_MARGIN = 0.08

# Classroom photos: slightly looser, always pick closest match per face
ATTENDANCE_THRESHOLD = 0.55


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def _parse_embedding(embedding):
    if embedding is None:
        return None
    try:
        arr = np.asarray(embedding, dtype=np.float64).flatten()
    except (TypeError, ValueError):
        return None
    if arr.size != 128:
        return None
    return arr


def get_face_database():
    """Load enrolled face embeddings from DB (always fresh)."""
    X = []
    y = []

    for student in get_all_students():
        emb = _parse_embedding(student.get("face_embedding"))
        if emb is not None:
            X.append(emb)
            y.append(int(student["student_id"]))

    return X, y


def get_face_embeddings(image_np, upsample=1):
    detector, sp, facerec = load_dlib_models()
    faces = detector(image_np, upsample)

    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(np.array(face_descriptor))
    return encodings


def _rank_distances(encoding, X, y):
    return sorted(
        [(int(y[i]), float(np.linalg.norm(X[i] - encoding))) for i in range(len(X))],
        key=lambda item: item[1],
    )


def match_face_login(encoding, X, y):
    """Strict match for student login (reject ambiguous faces)."""
    if not X:
        return None

    distances = _rank_distances(encoding, X, y)
    best_id, best_dist = distances[0]

    if best_dist > LOGIN_THRESHOLD:
        return None

    if len(distances) > 1:
        second_dist = distances[1][1]
        if (second_dist - best_dist) < LOGIN_MIN_MARGIN:
            return None

    return best_id


def match_face_attendance(encoding, X, y):
    """Match each face in a class photo to the closest enrolled student."""
    if not X:
        return None

    distances = _rank_distances(encoding, X, y)
    best_id, best_dist = distances[0]

    if best_dist <= ATTENDANCE_THRESHOLD:
        return best_id
    return None


def train_classifier():
    st.cache_resource.clear()
    return bool(get_face_database()[0])


def predict_student_login(class_image_np):
    """Single-user login scan."""
    encodings = get_face_embeddings(class_image_np, upsample=1)
    detected_student = {}
    X, y = get_face_database()
    all_students = sorted(set(y))

    if not X:
        return detected_student, all_students, len(encodings)

    for encoding in encodings:
        student_id = match_face_login(encoding, X, y)
        if student_id is not None:
            detected_student[student_id] = True

    return detected_student, all_students, len(encodings)


def predict_attendance(class_image_np):
    """Scan a classroom photo — detect every enrolled face present."""
    encodings = get_face_embeddings(class_image_np, upsample=2)
    detected_student = {}
    X, y = get_face_database()
    all_students = sorted(set(y))

    if not X:
        return detected_student, all_students, len(encodings)

    for encoding in encodings:
        student_id = match_face_attendance(encoding, X, y)
        if student_id is not None:
            detected_student[student_id] = True

    return detected_student, all_students, len(encodings)
