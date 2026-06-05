from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import io
import librosa
import streamlit as st

# Cosine similarity on normalized resemblyzer embeddings
VOICE_MATCH_THRESHOLD = 0.62
# Must be within this margin of the strongest match (reduces false positives)
VOICE_RELATIVE_MARGIN = 0.08


@st.cache_resource
def load_voice_encoder():
    return VoiceEncoder()


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float64).flatten()
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


def _parse_voice_embedding(embedding):
    if embedding is None:
        return None
    try:
        arr = np.asarray(embedding, dtype=np.float64).flatten()
    except (TypeError, ValueError):
        return None
    if arr.size < 64 or np.linalg.norm(arr) < 1e-6:
        return None
    return arr


def _similarity(query_emb, stored_emb):
    return float(np.dot(_normalize(query_emb), _normalize(stored_emb)))


def has_valid_voice_profile(embedding):
    return _parse_voice_embedding(embedding) is not None


def get_voice_embedding(audio_bytes):
    try:
        encoder = load_voice_encoder()
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        wav = preprocess_wav(audio)
        embedding = encoder.embed_utterance(wav)
        return embedding.tolist()
    except Exception:
        st.error('Voice recog error')
        return None


def _collect_query_embeddings(encoder, audio, sr):
    """Build several embeddings from one recording (full clip, segments, windows)."""
    queries = []

    try:
        queries.append(encoder.embed_utterance(preprocess_wav(audio)))
    except Exception:
        pass

    for start, end in librosa.effects.split(audio, top_db=20):
        if (end - start) < int(sr * 0.35):
            continue
        try:
            queries.append(
                encoder.embed_utterance(preprocess_wav(audio[start:end]))
            )
        except Exception:
            continue

    window = int(1.5 * sr)
    hop = int(0.75 * sr)
    for start in range(0, max(1, len(audio) - window), hop):
        chunk = audio[start : start + window]
        if len(chunk) < int(0.5 * sr):
            continue
        try:
            queries.append(encoder.embed_utterance(preprocess_wav(chunk)))
        except Exception:
            continue

    return queries


def process_bulk_audio(audio_bytes, candidates_dict, threshold=VOICE_MATCH_THRESHOLD):
    """
    Score every enrolled student independently (not winner-takes-all per segment).
    Returns {student_id: best_similarity} for all students with voice profiles.
    """
    try:
        parsed = {}
        for sid, emb in candidates_dict.items():
            vec = _parse_voice_embedding(emb)
            if vec is not None:
                parsed[int(sid)] = vec

        if not parsed:
            return {}

        encoder = load_voice_encoder()
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        if len(audio) < int(sr * 0.3):
            return {sid: 0.0 for sid in parsed}

        queries = _collect_query_embeddings(encoder, audio, sr)
        if not queries:
            return {sid: 0.0 for sid in parsed}

        student_scores = {sid: 0.0 for sid in parsed}

        for query_emb in queries:
            for sid, stored in parsed.items():
                sim = _similarity(query_emb, stored)
                if sim > student_scores[sid]:
                    student_scores[sid] = sim

        return student_scores
    except Exception:
        st.error('Bulk process error')
        return {}


def classify_voice_presence(
    student_scores,
    threshold=VOICE_MATCH_THRESHOLD,
    relative_margin=VOICE_RELATIVE_MARGIN,
):
    """
    Mark present only if similarity is high enough AND close to the top match.
    Stops marking silent students who only get weak background similarity.
    """
    if not student_scores:
        return {}

    max_score = max(student_scores.values())
    if max_score < threshold:
        return {sid: False for sid in student_scores}

    cutoff = max(threshold, max_score - relative_margin)
    return {sid: score >= cutoff for sid, score in student_scores.items()}
