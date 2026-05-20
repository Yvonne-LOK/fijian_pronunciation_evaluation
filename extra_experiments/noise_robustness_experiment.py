"""
噪声稳健性对比实验：GFCC-DTW vs MFCC-DTW

对 30 条 Fijian 母语朗读句子，在 clean / 20 dB / 10 dB / 5 dB 四种 SNR 下，
分别用 GFCC-DTW 和 MFCC-DTW 计算句子级评分，比较两种特征的噪声稳健性。

输出：
  - 加噪音频文件：experiments/noise_robustness/audio/snr{X}/
  - 明细评分 CSV：experiments/noise_robustness/results/scores_detail.csv
  - 汇总指标 CSV：experiments/noise_robustness/results/summary.csv
  - 对比图：experiments/noise_robustness/results/plots/*.png
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.fftpack import dct
from dtw import dtw
from gammatone.gtgram import gtgram


# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------
SR = 16000
WIN_TIME = 0.025
HOP_TIME = 0.010
N_FILTERS = 32
N_CEPS = 13
F_MIN = 50

SNR_LIST_DB = [None, 20, 10, 5]  # None = clean
N_SENTENCES = 30
SEED_SAMPLE = 42
SEED_NOISE = 2026

ALPHA = 8.0
BETA = 0.6

SRC_DIR = Path(
    r"D:\北外\2024-26 研究生毕设\调研\北外多语语音评测\static\fijian"
)
OUT_ROOT = Path(r"D:\workdir\experiments\noise_robustness")
AUDIO_DIR = OUT_ROOT / "audio"
RESULT_DIR = OUT_ROOT / "results"
PLOT_DIR = RESULT_DIR / "plots"


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def load_wav(wav_path: Path, target_sr: int = SR):
    y, sr = sf.read(str(wav_path))
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    max_abs = np.max(np.abs(y)) + 1e-8
    y = y / max_abs
    return y.astype(np.float32), sr


def add_gaussian_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator):
    """加入高斯白噪声到指定 SNR (dB)"""
    sig_power = float(np.mean(x ** 2))
    if sig_power <= 0:
        return x.copy()
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(x)).astype(np.float32) * np.sqrt(noise_power)
    return (x + noise).astype(np.float32)


def z_norm(feats: np.ndarray):
    mean = np.mean(feats, axis=0, keepdims=True)
    std = np.std(feats, axis=0, keepdims=True) + 1e-8
    return (feats - mean) / std


# ------------------------------------------------------------------
# GFCC（与原 gfcc_dtw_score.py 完全一致）
# ------------------------------------------------------------------
def extract_gfcc(y: np.ndarray, sr: int):
    gram = gtgram(
        wave=y,
        fs=sr,
        window_time=WIN_TIME,
        hop_time=HOP_TIME,
        channels=N_FILTERS,
        f_min=F_MIN,
    )
    gram = np.log(gram + 1e-8)
    ceps = dct(gram, type=2, axis=0, norm="ortho")[:N_CEPS, :]
    return ceps.T.astype(np.float32)


# ------------------------------------------------------------------
# MFCC（对称实现：mel filterbank + log + DCT，路径与 GFCC 一致）
# ------------------------------------------------------------------
def extract_mfcc(y: np.ndarray, sr: int):
    n_fft = int(round(WIN_TIME * sr))   # 25ms 窗 → 400 点 @16kHz
    hop_length = int(round(HOP_TIME * sr))  # 10ms 移 → 160 点

    # 短时傅里叶 → 功率谱
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=False,
    )
    power = np.abs(stft) ** 2  # (n_fft/2+1, T)

    # mel filterbank：与 GFCC 通道数对齐
    mel_fb = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=N_FILTERS,
        fmin=F_MIN,
        fmax=sr / 2,
    )  # (N_FILTERS, n_fft/2+1)

    mel_energy = mel_fb @ power  # (N_FILTERS, T)
    mel_energy = np.log(mel_energy + 1e-8)

    ceps = dct(mel_energy, type=2, axis=0, norm="ortho")[:N_CEPS, :]
    return ceps.T.astype(np.float32)


# ------------------------------------------------------------------
# DTW 评分
# ------------------------------------------------------------------
def compute_dtw_score(feat_ref: np.ndarray, feat_test: np.ndarray):
    alignment = dtw(feat_ref, feat_test, dist_method="euclidean")
    raw_d = float(alignment.distance)
    path_len = len(alignment.index1)
    norm_d = raw_d / max(path_len, 1)
    score = 100.0 * np.exp(-ALPHA * norm_d / max(BETA, 1e-8))
    score = float(np.clip(score, 0.0, 100.0))
    return norm_d, score


def score_pair(y_ref: np.ndarray, y_test: np.ndarray, feature: str):
    if feature == "gfcc":
        f_ref = extract_gfcc(y_ref, SR)
        f_test = extract_gfcc(y_test, SR)
    elif feature == "mfcc":
        f_ref = extract_mfcc(y_ref, SR)
        f_test = extract_mfcc(y_test, SR)
    else:
        raise ValueError(feature)
    f_ref = z_norm(f_ref)
    f_test = z_norm(f_test)
    return compute_dtw_score(f_ref, f_test)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def select_sentences():
    wavs = sorted(SRC_DIR.glob("u1_l1_*.wav"))
    rng = random.Random(SEED_SAMPLE)
    picked = rng.sample(wavs, N_SENTENCES)
    picked.sort()  # 按文件名重排，便于阅读
    return picked


def snr_tag(snr_db):
    return "clean" if snr_db is None else f"snr{int(snr_db):02d}"


def prepare_audio(picked):
    """生成加噪音频并保存"""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for snr_db in SNR_LIST_DB:
        (AUDIO_DIR / snr_tag(snr_db)).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED_NOISE)
    rows = []
    for src in picked:
        y, _ = load_wav(src)
        for snr_db in SNR_LIST_DB:
            tag = snr_tag(snr_db)
            out_path = AUDIO_DIR / tag / src.name
            if snr_db is None:
                y_out = y
            else:
                y_out = add_gaussian_noise(y, float(snr_db), rng)
                # 防止溢出
                m = np.max(np.abs(y_out))
                if m > 1.0:
                    y_out = y_out / m
            sf.write(str(out_path), y_out, SR, subtype="PCM_16")
            rows.append(
                {
                    "sentence_id": src.stem,
                    "snr": tag,
                    "src": str(src),
                    "noisy_path": str(out_path),
                }
            )
    return pd.DataFrame(rows)


def run_scoring(picked):
    rows = []
    for i, src in enumerate(picked, 1):
        print(f"[{i:2d}/{len(picked)}] {src.name}", flush=True)
        y_ref, _ = load_wav(src)
        for snr_db in SNR_LIST_DB:
            tag = snr_tag(snr_db)
            test_path = AUDIO_DIR / tag / src.name
            y_test, _ = load_wav(test_path)
            for feature in ("gfcc", "mfcc"):
                norm_d, score = score_pair(y_ref, y_test, feature)
                rows.append(
                    {
                        "sentence_id": src.stem,
                        "feature": feature,
                        "snr": tag,
                        "norm_distance": norm_d,
                        "score": score,
                    }
                )
    return pd.DataFrame(rows)


def summarize(detail_df: pd.DataFrame):
    # clean 分数（参考基线）
    clean = (
        detail_df[detail_df["snr"] == "clean"]
        .groupby("feature")["score"]
        .agg(clean_mean="mean", clean_std="std")
        .reset_index()
    )

    # 噪声分数与衰减
    rows = []
    for feature in ("gfcc", "mfcc"):
        clean_mean = clean.loc[clean["feature"] == feature, "clean_mean"].iloc[0]
        for tag in ("clean", "snr20", "snr10", "snr05"):
            sub = detail_df[
                (detail_df["feature"] == feature) & (detail_df["snr"] == tag)
            ]["score"]
            rows.append(
                {
                    "feature": feature,
                    "snr": tag,
                    "mean_score": float(sub.mean()),
                    "std_score": float(sub.std()),
                    "mean_drop": float(clean_mean - sub.mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_results(summary_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    snr_order = ["snr20", "snr10", "snr05"]
    snr_label = ["20 dB", "10 dB", "5 dB"]
    x = np.arange(len(snr_order))

    # 图1：mean_drop vs SNR
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=140)
    for feat, marker, color in [("gfcc", "o", "#1f77b4"), ("mfcc", "s", "#d62728")]:
        ys = [
            summary_df[
                (summary_df["feature"] == feat) & (summary_df["snr"] == s)
            ]["mean_drop"].iloc[0]
            for s in snr_order
        ]
        ax.plot(x, ys, marker=marker, linewidth=2, label=feat.upper(), color=color)
        for xi, yi in zip(x, ys):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(snr_label)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Mean Score Drop (clean − noisy)")
    ax.set_title("Score degradation under additive white noise")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "mean_drop_vs_snr.png"
    fig.savefig(p1)
    plt.close(fig)

    # 图2：std vs SNR
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=140)
    for feat, marker, color in [("gfcc", "o", "#1f77b4"), ("mfcc", "s", "#d62728")]:
        ys = [
            summary_df[
                (summary_df["feature"] == feat) & (summary_df["snr"] == s)
            ]["std_score"].iloc[0]
            for s in snr_order
        ]
        ax.plot(x, ys, marker=marker, linewidth=2, label=feat.upper(), color=color)
        for xi, yi in zip(x, ys):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(snr_label)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Score Std. across 30 sentences")
    ax.set_title("Score stability under additive white noise")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "std_vs_snr.png"
    fig.savefig(p2)
    plt.close(fig)

    return p1, p2


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> Step 1: select 30 sentences")
    picked = select_sentences()
    with open(RESULT_DIR / "selected_sentences.txt", "w", encoding="utf-8") as fh:
        for p in picked:
            fh.write(p.name + "\n")
    print(f"selected {len(picked)} sentences")

    print(">>> Step 2: generate noisy audio")
    audio_index = prepare_audio(picked)
    audio_index.to_csv(RESULT_DIR / "audio_index.csv", index=False)

    print(">>> Step 3: GFCC + MFCC scoring on 120 pairs")
    detail = run_scoring(picked)
    detail.to_csv(RESULT_DIR / "scores_detail.csv", index=False)

    print(">>> Step 4: summarize")
    summary = summarize(detail)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))

    print(">>> Step 5: plot")
    p1, p2 = plot_results(summary, PLOT_DIR)
    print(f"saved {p1}\nsaved {p2}")

    print("\n>>> done")


if __name__ == "__main__":
    main()
