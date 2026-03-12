import json
import torch
import librosa
import numpy as np
from tqdm import tqdm


def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)


#(SpeechBrain x-vector)
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition

speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)

def get_speaker_embedding(wav_path):
    signal = load_audio(wav_path)
    signal_tensor = torch.tensor(signal).unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        emb = speaker_model.encode_file(wav_path)  # 1 x embedding_dim
    return emb.squeeze().cpu().numpy()

import openl3

def get_acoustic_embedding(wav_path):
    wav, sr = librosa.load(wav_path, sr=None)
    emb, ts = openl3.get_audio_embedding(
        wav, sr,
        model="music", embedding_size=512
    )
    return emb.mean(axis=0)  # 对时间取平均

# -----------------------------
# 4. 主流程
# -----------------------------
def compute_speaker_acoustic_cosine(data):

    speaker_sims = []
    acoustic_sims = []

    for s in tqdm(data):
        gen_audio = s["audio"]
        ref_audio = s["ref_audio"]

        # 说话人 embedding
        spk_ref = get_speaker_embedding(ref_audio)
        spk_gen = get_speaker_embedding(gen_audio)
        speaker_sims.append(cosine_similarity(spk_ref, spk_gen))

        # 声学 embedding
        ac_ref = get_acoustic_embedding(ref_audio)
        ac_gen = get_acoustic_embedding(gen_audio)
        acoustic_sims.append(cosine_similarity(ac_ref, ac_gen))

    result = {
        "speaker_cosine_mean": float(np.mean(speaker_sims)),
        "acoustic_cosine_mean": float(np.mean(acoustic_sims)),
        "num_samples": len(data)
    }

    return result

