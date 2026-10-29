"""Deterministic physical concepts: six for Node1, five for Node2, and three for Node3."""

from __future__ import annotations
import numpy as np
from scipy import ndimage, signal

NODE1_FEATURE_NAMES = (
    "H_ridge_prom_med",
    "H_ridge_prom_q25",
    "H_ridge_bandwidth_ratio",
    "H_ms_zmax",
    "H_ms_center_consistency",
    "O_anisotropic_response_ratio",
)
NODE2_FEATURE_NAMES = (
    "V_z",
    "V_delta_db",
    "V_top5_z_mean",
    "V_longest_run_z3",
    "P_envelope_autocorr_secondary",
)
NODE3_FEATURE_NAMES = (
    "PC_envelope_ac_lag_norm",
    "STFT_time_top10_concentration",
    "IQ_complex_ac_lag_norm",
)
MID_FREQ_START = 30
MID_FREQ_END = 226
RIDGE_HALF_WIDTH = 2
MULTISCALE_WIDTHS = (3, 5, 9, 17, 25)
TIME_BLOCKS = 8
PROFILE_BINS = 512
EPS_NODE1 = 1e-06
EPS_NODE2 = 1e-06
EPS_NODE2_PULSE = 1e-08
EPS_NODE3 = 1e-09


def normalize_stft(stft_db: np.ndarray) -> np.ndarray:
    image = np.asarray(stft_db, dtype=np.float32)
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum > minimum:
        return ((image - minimum) / (maximum - minimum)).astype(np.float32)
    return (image - minimum).astype(np.float32)


def quantize_profiles(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(np.float16).astype(np.float32)


def _crop_midband(stft_db: np.ndarray) -> np.ndarray:
    image = np.asarray(stft_db, dtype=np.float64)
    if image.ndim != 2 or image.shape[1] == 0:
        raise ValueError(f"Expected a non-empty 2-D STFT image, got {image.shape}")
    start = min(MID_FREQ_START, image.shape[0])
    end = min(MID_FREQ_END, image.shape[0])
    if end <= start:
        raise ValueError(f"Cannot crop [{start}:{end}] from STFT {image.shape}")
    return image[start:end]


def _half_prominence_width(profile: np.ndarray, peak_index: int) -> int:
    values = np.asarray(profile, dtype=np.float64).reshape(-1)
    floor = float(np.median(values))
    peak = float(values[peak_index])
    prominence = peak - floor
    if not np.isfinite(prominence) or prominence <= 1e-12:
        return int(values.size)
    level = floor + 0.5 * prominence
    left = right = int(peak_index)
    while left > 0 and values[left - 1] >= level:
        left -= 1
    while right + 1 < values.size and values[right + 1] >= level:
        right += 1
    return int(right - left + 1)


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)) + EPS_NODE1)
    return (center, scale)


def _uniform_filter_1d(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if size <= 1:
        return values.copy()
    half = size // 2
    padded = np.pad(values, (half, half), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    return (cumulative[size:] - cumulative[:-size]) / float(size)


def _contrast_profile(profile: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    band = _uniform_filter_1d(values, width)
    outer_width = 3 * width
    outer = _uniform_filter_1d(values, outer_width)
    side = (outer * outer_width - band * width) / max(1, outer_width - width)
    contrast = band - side
    margin = min(len(values) // 2, 2 * width)
    if margin:
        contrast[:margin] = -np.inf
        contrast[-margin:] = -np.inf
    return contrast


def _best_multiscale_candidate(profile: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for width in MULTISCALE_WIDTHS:
        contrast = _contrast_profile(profile, width)
        finite = np.isfinite(contrast)
        if not np.any(finite):
            continue
        baseline, scale = _robust_scale(contrast[finite])
        index = int(np.argmax(contrast))
        delta = float(contrast[index])
        candidate: dict[str, float | int] = {
            "center": index,
            "width": int(width),
            "delta": delta,
            "z": float((delta - baseline) / scale),
        }
        if best is None or (float(candidate["z"]), delta) > (
            float(best["z"]),
            float(best["delta"]),
        ):
            best = candidate
    return best or {"center": 0, "width": 1, "delta": 0.0, "z": 0.0}


def _center_consistency(mid: np.ndarray, center: int, width: int) -> float:
    center_hits = []
    for block in np.array_split(mid, TIME_BLOCKS, axis=1):
        if block.shape[1] == 0:
            continue
        block_profile = np.median(block, axis=1)
        block_best = _best_multiscale_candidate(block_profile)
        center_hits.append(
            abs(int(block_best["center"]) - int(center)) <= max(2, width)
        )
    return float(np.mean(center_hits)) if center_hits else 0.0


def extract_node1(stft_db: np.ndarray) -> tuple[np.ndarray, np.float32]:
    image = np.asarray(stft_db, dtype=np.float64)
    mid = _crop_midband(image)
    row_median = np.median(mid, axis=1)
    dominant_local = int(np.argmax(row_median))
    dominant = dominant_local + min(MID_FREQ_START, image.shape[0])
    start = min(MID_FREQ_START, image.shape[0])
    end = min(MID_FREQ_END, image.shape[0])
    lo = max(start, dominant - RIDGE_HALF_WIDTH)
    hi = min(end, dominant + RIDGE_HALF_WIDTH + 1)
    per_time_floor = np.median(mid, axis=0)
    ridge_prominence = np.mean(image[lo:hi], axis=0) - per_time_floor
    q25_profile = np.percentile(mid, 25.0, axis=1)
    candidate = _best_multiscale_candidate(q25_profile)
    center_consistency = _center_consistency(
        mid, center=int(candidate["center"]), width=int(candidate["width"])
    )
    z = np.clip(
        (mid - float(np.median(mid)))
        / (float(1.4826 * np.median(np.abs(mid - float(np.median(mid))))) + EPS_NODE1),
        -4.0,
        10.0,
    )
    horizontal = ndimage.gaussian_filter(
        z, sigma=(1.0, 6.0), mode="nearest"
    ) - ndimage.gaussian_filter(z, sigma=(4.0, 6.0), mode="nearest")
    vertical = ndimage.gaussian_filter(
        z, sigma=(6.0, 1.0), mode="nearest"
    ) - ndimage.gaussian_filter(z, sigma=(6.0, 4.0), mode="nearest")
    horizontal_energy = float(np.mean(np.abs(horizontal)))
    vertical_energy = float(np.mean(np.abs(vertical)))
    anisotropic_ratio = horizontal_energy / (
        horizontal_energy + vertical_energy + EPS_NODE1
    )
    values = np.asarray(
        [
            float(np.median(ridge_prominence)),
            float(np.percentile(ridge_prominence, 25.0)),
            float(
                _half_prominence_width(row_median, dominant_local)
                / max(1, len(row_median))
            ),
            float(candidate["z"]),
            center_consistency,
            anisotropic_ratio,
        ],
        dtype=np.float32,
    )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    med, q25, bandwidth, ms_zmax, center_consistency, anisotropic = values.astype(
        np.float64
    )
    high_positive = q25 >= 0.5
    high_negative = q25 <= -0.5
    if high_positive:
        v1_probability = 0.75 + 0.25 * np.tanh((q25 - 0.5) / 2.0)
    elif high_negative:
        v1_probability = 0.25 - 0.25 * np.tanh((-0.5 - q25) / 2.0)
    else:
        v1_probability = 1.0 / (1.0 + np.exp(-(bandwidth - 0.04) / 0.01))
    margins = np.asarray(
        [
            (q25 - -1.75) / 0.5,
            (0.5 - q25) / 0.5,
            (med - 0.6) / 0.5,
            ms_zmax - 2.5,
            (center_consistency - 0.125) / 0.125,
            (anisotropic - 0.53) / 0.02,
        ],
        dtype=np.float64,
    )
    weak_margin = float(np.min(margins))
    weak_probability = 1.0 / (1.0 + np.exp(-np.clip(weak_margin, -30.0, 30.0)))
    probability = np.float32(max(float(v1_probability), weak_probability))
    return (values, probability)


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool).reshape(-1):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _node2_pulse_profile(envelope: np.ndarray) -> np.ndarray:
    values = np.asarray(envelope, dtype=np.float64)
    maximum = float(np.max(values))
    normalized = values / (maximum + EPS_NODE2_PULSE)
    usable = PROFILE_BINS * (len(normalized) // PROFILE_BINS)
    if usable < PROFILE_BINS:
        grid = np.linspace(0, len(normalized) - 1, PROFILE_BINS)
        compact = np.interp(grid, np.arange(len(normalized)), normalized)
        maximum_profile = compact
        mean_profile = compact
    else:
        offset = (len(normalized) - usable) // 2
        windows = normalized[offset : offset + usable].reshape(PROFILE_BINS, -1)
        maximum_profile = np.max(windows, axis=1)
        mean_profile = np.mean(windows, axis=1)
    return np.stack(
        [
            np.log1p(20.0 * maximum_profile) / np.log(21.0),
            np.log1p(20.0 * mean_profile) / np.log(21.0),
        ],
        axis=0,
    ).astype(np.float32)


def _secondary_autocorrelation(envelope: np.ndarray) -> float:
    values = np.asarray(envelope, dtype=np.float64)
    values = values - np.mean(values)
    if not np.any(values):
        return 0.0
    size = 1 << int(np.ceil(np.log2(2 * len(values) - 1)))
    spectrum = np.fft.rfft(values, n=size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[: len(values)]
    zero = float(abs(autocorrelation[0]))
    start = max(8, len(values) // 500)
    end = max(start + 1, len(values) // 2)
    return float(np.max(np.abs(autocorrelation[start:end])) / (zero + EPS_NODE2_PULSE))


def extract_node2(
    stft_db: np.ndarray, b_pulse: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    image = np.asarray(stft_db, dtype=np.float64)
    start = min(MID_FREQ_START, image.shape[0])
    end = min(MID_FREQ_END, image.shape[0])
    residual = image - np.median(image, axis=1, keepdims=True)
    positive = np.maximum(residual[start:end], 0.0)
    column_energy = np.mean(positive, axis=0)
    baseline = float(np.median(column_energy))
    robust_sigma = float(
        1.4826 * np.median(np.abs(column_energy - baseline)) + EPS_NODE2
    )
    z_curve = (column_energy - baseline) / robust_sigma
    peak = float(np.max(column_energy))
    above3 = z_curve >= 3.0
    top_count = max(1, int(np.ceil(0.05 * len(z_curve))))
    top5_z_mean = float(np.mean(np.partition(z_curve, -top_count)[-top_count:]))
    positive_excess = np.maximum(column_energy - baseline, 0.0)
    temporal_profile = np.stack(
        [
            np.clip(z_curve, -3.0, 30.0) / 10.0,
            np.clip(positive_excess, 0.0, 12.0) / 3.0,
        ],
        axis=0,
    ).astype(np.float32)
    envelope = np.abs(np.asarray(b_pulse)).astype(np.float64)
    pulse_autocorrelation = _secondary_autocorrelation(envelope)
    pulse_profile = _node2_pulse_profile(envelope)
    concepts = np.asarray(
        [
            float(np.max(z_curve)),
            peak - baseline,
            top5_z_mean,
            float(_longest_true_run(above3) / len(z_curve)),
            pulse_autocorrelation,
        ],
        dtype=np.float32,
    )
    concepts = np.nan_to_num(concepts, nan=0.0, posinf=0.0, neginf=0.0)
    strong = min((float(concepts[0]) - 10.0) / 2.0, (float(concepts[1]) - 3.0) / 0.5)
    persistent = min(float(concepts[2]) - 4.0, (float(concepts[3]) - 0.02) / 0.01)
    score = np.asarray([max(strong, persistent)], dtype=np.float32)
    logit = np.clip(score / 0.5, -12.0, 12.0)
    probability = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)[0]
    return (concepts, temporal_profile, pulse_profile, probability)


def _autocorr_peak(
    values: np.ndarray, min_lag: int, max_lag: int
) -> tuple[float, float]:
    x = np.asarray(values)
    if x.size <= min_lag + 1:
        return (0.0, 0.0)
    x = x - np.mean(x)
    ac = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1 :]
    zero = float(np.abs(ac[0]))
    hi = min(int(max_lag), len(ac) - 1)
    lo = min(max(1, int(min_lag)), hi)
    if zero <= EPS_NODE3 or hi < lo:
        return (0.0, 0.0)
    segment = np.abs(ac[lo : hi + 1]) / (zero + EPS_NODE3)
    offset = int(np.argmax(segment))
    lag = lo + offset
    return (float(segment[offset]), float(lag / len(x)))


def _top_fraction_ratio(values: np.ndarray, fraction: float) -> float:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if values.size == 0:
        return 0.0
    total = float(values.sum())
    if total <= EPS_NODE3:
        return 0.0
    count = max(1, int(np.ceil(fraction * len(values))))
    return float(np.partition(values, -count)[-count:].sum() / total)


def _resample(values: np.ndarray, bins: int = PROFILE_BINS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    if values.size == bins:
        return values.astype(np.float32)
    source = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    target = np.linspace(0.0, 1.0, bins, dtype=np.float64)
    return np.interp(target, source, values).astype(np.float32)


def _unit_peak(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    if maximum > EPS_NODE3:
        return values / (maximum + EPS_NODE3)
    return np.zeros_like(values)


def _autocorrelation_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.mean(values)
    correlation = signal.correlate(values, values, mode="full", method="fft")[
        len(values) - 1 :
    ]
    zero = float(np.abs(correlation[0])) if correlation.size else 0.0
    if zero > EPS_NODE3:
        return np.abs(correlation) / (zero + EPS_NODE3)
    return np.zeros_like(correlation)


def extract_pc_profile(b_pulse: np.ndarray) -> np.ndarray:
    amplitude = _unit_peak(np.abs(np.asarray(b_pulse)))
    smooth = ndimage.uniform_filter1d(amplitude, size=16, mode="nearest")[::4]
    autocorrelation = _autocorrelation_profile(smooth)
    return np.stack([_resample(smooth), _resample(autocorrelation)], axis=0).astype(
        np.float32
    )


def extract_iq_profile(c_pulse: np.ndarray) -> np.ndarray:
    iq = np.asarray(c_pulse)
    amplitude = ndimage.uniform_filter1d(
        np.abs(iq).astype(np.float64), size=32, mode="nearest"
    )[::4]
    amplitude = _unit_peak(amplitude)
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) if iq.size else 0.0
    normalized = iq / (rms + EPS_NODE3) if rms > EPS_NODE3 else np.zeros_like(iq)
    return np.stack(
        [
            _resample(amplitude),
            _resample(np.real(normalized)),
            _resample(np.imag(normalized)),
        ],
        axis=0,
    ).astype(np.float32)


def extract_node3(
    c_pulse: np.ndarray, b_pulse: np.ndarray, stft_db: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    amplitude = np.abs(np.asarray(b_pulse)).astype(np.float64)
    maximum = float(np.max(amplitude)) if len(amplitude) else 0.0
    if len(amplitude) == 0 or maximum <= EPS_NODE3:
        pc_lag = 0.0
    else:
        normalized = amplitude / (maximum + EPS_NODE3)
        smooth = ndimage.uniform_filter1d(normalized, size=16, mode="nearest")[::4]
        _, pc_lag = _autocorr_peak(smooth, min_lag=8, max_lag=len(smooth) // 2)
    image = np.asarray(stft_db, dtype=np.float64)
    if image.ndim != 2 or image.size == 0:
        concentration = 0.0
    else:
        power = 10.0 ** ((image - np.max(image)) / 10.0)
        curve = np.sum(power, axis=0)
        baseline = float(np.median(curve))
        excess = np.maximum(curve - baseline, 0.0)
        concentration = _top_fraction_ratio(excess, 0.1)
    iq = np.asarray(c_pulse)
    if iq.size == 0 or float(np.max(np.abs(iq))) <= EPS_NODE3:
        iq_lag = 0.0
    else:
        _, iq_lag = _autocorr_peak(iq, min_lag=32, max_lag=len(iq) // 2)
    concepts = np.nan_to_num(
        np.asarray([pc_lag, concentration, iq_lag], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    primary = (0.05 - float(concepts[0])) / 0.02
    fallback = min((0.05 - float(concepts[2])) / 0.02, (float(concepts[1]) - 0.8) / 0.1)
    score = np.asarray([max(primary, fallback)], dtype=np.float32)
    logit = np.clip(score / 0.5, -12.0, 12.0)
    probability = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)[0]
    return (
        concepts,
        extract_pc_profile(b_pulse),
        extract_iq_profile(c_pulse),
        probability,
    )
