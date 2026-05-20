# 单对音频片段评分

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from scipy.fftpack import dct
from dtw import dtw
from gammatone.gtgram import gtgram


def load_wav(wav_path: str, target_sr: int = 16000):
    """
    读取音频，转单声道，重采样到 target_sr
    """
    y, sr = sf.read(wav_path)

    # 若为双声道，转单声道
    if y.ndim == 2:
        y = np.mean(y, axis=1)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 振幅归一化，避免极端数值
    max_abs = np.max(np.abs(y)) + 1e-8
    y = y / max_abs

    return y.astype(np.float32), sr


def extract_gfcc(
    y: np.ndarray,
    sr: int,
    win_time: float = 0.025,
    hop_time: float = 0.010,
    n_filters: int = 32,
    n_ceps: int = 13,
    f_min: int = 50,
):
    """
    用 gammatone filterbank 近似实现 GFCC:
    1) gtgram 得到 gammatone 频带能量
    2) log 压缩
    3) DCT 得到 cepstral coefficients
    输出形状: (T, n_ceps)
    """
    # gtgram 参数
    # window_time, hop_time, channels, f_min
    gram = gtgram(
        wave=y,
        fs=sr,
        window_time=win_time,
        hop_time=hop_time,
        channels=n_filters,
        f_min=f_min,
    )  # shape: (n_filters, T)

    # 对数压缩
    gram = np.log(gram + 1e-8)

    # DCT 到倒谱域
    ceps = dct(gram, type=2, axis=0, norm="ortho")[:n_ceps, :]  # (n_ceps, T)

    # 转成 (T, n_ceps)
    feats = ceps.T.astype(np.float32)
    return feats


def z_norm(feats: np.ndarray):
    """
    按维度做 z-normalization
    """
    mean = np.mean(feats, axis=0, keepdims=True)
    std = np.std(feats, axis=0, keepdims=True) + 1e-8
    return (feats - mean) / std


def compute_dtw_distance(feat_ref: np.ndarray, feat_test: np.ndarray):
    """
    计算 DTW 距离，使用欧氏距离
    返回：
      raw_distance: 原始累计距离
      norm_distance: 按路径长度归一化后的距离
      path_len: DTW 路径长度
    """
    alignment = dtw(
        feat_ref,
        feat_test,
        dist_method="euclidean",
    )

    raw_distance = float(alignment.distance)
    path_len = len(alignment.index1)
    norm_distance = raw_distance / max(path_len, 1)

    return raw_distance, norm_distance, path_len


def distance_to_score(
    norm_distance: float,
    alpha: float = 8.0,
    beta: float = 0.6,
):
    """
    将归一化距离映射到 0-100 分
    分数越高越好

    score = 100 * exp(-alpha * d / beta) 的一种变体
    这里用更平滑的指数衰减
    """
    score = 100.0 * np.exp(-alpha * norm_distance / max(beta, 1e-8))
    score = float(np.clip(score, 0.0, 100.0))
    return score


def score_two_segments(ref_wav: str, test_wav: str):
    """
    对两段音频做 GFCC + DTW 评分
    """
    y_ref, sr_ref = load_wav(ref_wav, target_sr=16000)
    y_test, sr_test = load_wav(test_wav, target_sr=16000)

    ref_feat = extract_gfcc(y_ref, sr_ref)
    test_feat = extract_gfcc(y_test, sr_test)

    ref_feat = z_norm(ref_feat)
    test_feat = z_norm(test_feat)

    raw_d, norm_d, path_len = compute_dtw_distance(ref_feat, test_feat)
    score = distance_to_score(norm_d)

    return {
        "ref_wav": ref_wav,
        "test_wav": test_wav,
        "ref_num_frames": int(ref_feat.shape[0]),
        "test_num_frames": int(test_feat.shape[0]),
        "feat_dim": int(ref_feat.shape[1]),
        "dtw_raw_distance": raw_d,
        "dtw_norm_distance": norm_d,
        "dtw_path_len": path_len,
        "score": score,
    }


def main():
    # 这里先拿两段同一个词测试
    ref_wav = "data/fijian/parsed/words/u1_l1_008/001_taubale.wav"
    # test_wav = "data/fijian/parsed/words/u1_l1_008/001_taubale.wav"
    test_wav = "data/fijian/parsed/words/u1_l1_008/004_tagane.wav"

    result = score_two_segments(ref_wav, test_wav)

    print("=== GFCC + DTW scoring result ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()