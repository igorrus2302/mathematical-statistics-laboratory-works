from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy import stats

# начальные условия
MU1 = 5.0
MU2 = 5.0
SIGMA2_1 = 2.0  # дисперсия X1
SIGMA2_2 = 2.0  # дисперсия X2
SIGMA_1 = math.sqrt(SIGMA2_1)
SIGMA_2 = math.sqrt(SIGMA2_2)
N1 = 100
N2 = 100

# уровень значимости по умолчанию
ALPHA = 0.05
RNG_SEED = 2025

# вспомогательные функции
def two_sided_p_from_stat_cdf(stat: float, cdf_func) -> float:
    p_left = cdf_func(stat)
    p_right = 1.0 - p_left
    return 2.0 * min(p_left, p_right)


def f_test_two_sided(ratio: float, dfn: int, dfd: int) -> float:
    if ratio < 1:
        ratio = 1.0 / ratio
        dfn, dfd = dfd, dfn
    # Односторонняя верхняя хвостовая вероятность
    p_one_tail = stats.f.sf(ratio, dfn, dfd)
    return 2.0 * min(p_one_tail, 1.0)  # безопасно на случай числ. погрешностей


@dataclass
class SampleData:
    x1: np.ndarray
    x2: np.ndarray

    @property
    def n1(self) -> int:
        return self.x1.size

    @property
    def n2(self) -> int:
        return self.x2.size


# 1) Генерация данных и выборочные характеристики
def generate_data(seed: int = RNG_SEED) -> SampleData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(loc=MU1, scale=SIGMA_1, size=N1)
    x2 = rng.normal(loc=MU2, scale=SIGMA_2, size=N2)
    return SampleData(x1=x1, x2=x2)


def sample_characteristics(data: SampleData) -> Dict[str, Dict[str, float]]:
    """
    Возвращает табличку со средними/дисперсиями/СКО и размерами выборок.
    Используем stats.describe: дисперсия — несмещённая (ddof=1).
    """
    def desc(x: np.ndarray) -> Tuple[float, float, float, int]:
        d = stats.describe(x)
        mean = d.mean
        var = d.variance  # несмещённая
        std = math.sqrt(var)
        n = d.nobs
        return mean, var, std, n

    mean1, var1, std1, n1 = desc(data.x1)
    mean2, var2, std2, n2 = desc(data.x2)

    # Смешанная (pooled) дисперсия — для равных дисперсий
    s_p2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    out = {
        "X1": {"mean": mean1, "var": var1, "std": std1, "n": n1},
        "X2": {"mean": mean2, "var": var2, "std": std2, "n": n2},
        "Pooled": {"mean": np.nan, "var": s_p2, "std": math.sqrt(s_p2), "n": n1 + n2},
    }
    return out


# 2) Однопараметрические критерии (для X1)
def one_sample_tests(x: np.ndarray, alpha: float = ALPHA) -> pd.DataFrame:
    n = x.size
    xbar = float(np.mean(x))
    s2 = float(np.var(x, ddof=1))

    rows = []

    # z-test: H0: m = MU1 при известной σ^2 = SIGMA2_1
    z_stat = (xbar - MU1) / (SIGMA_1 / math.sqrt(n))
    z_p = two_sided_p_from_stat_cdf(z_stat, stats.norm.cdf)
    rows.append(["z‑test (H0: m = 5, σ^2 известна)", z_stat, z_p, z_p < alpha])

    # t-test: H0: m = MU1 при неизвестной σ
    t_stat, t_p = stats.ttest_1samp(x, popmean=MU1, alternative="two-sided")
    rows.append(["t‑test (H0: m = 5)", t_stat, t_p, t_p < alpha])

    # χ^2‑test (m известна): H0: σ^2 = 2 при известном m = 5
    chi_known = np.sum((x - MU1) ** 2) / SIGMA2_1

    # при m известна: χ^2 ~ Chi2(df = n)
    df_known = n
    p_left = stats.chi2.cdf(chi_known, df_known)
    p_right = stats.chi2.sf(chi_known, df_known)
    chi_known_p = 2.0 * min(p_left, p_right)
    rows.append(["χ²‑test (H0: σ² = 2, m известна)", chi_known, chi_known_p, chi_known_p < alpha])

    # χ^2‑test (m неизвестна): H0: σ^2 = 2 при неизвестном m
    chi_unknown = (n - 1) * s2 / SIGMA2_1
    df_unknown = n - 1
    p_left = stats.chi2.cdf(chi_unknown, df_unknown)
    p_right = stats.chi2.sf(chi_unknown, df_unknown)
    chi_unknown_p = 2.0 * min(p_left, p_right)
    rows.append(["χ²‑test (H0: σ² = 2, m неизвестна)", chi_unknown, chi_unknown_p, chi_unknown_p < alpha])

    df = pd.DataFrame(rows, columns=["Критерий", "Статистика", "p‑value", f"Решение при α={alpha}"])
    return df


# 3) Двухвыборочные критерии (X1 vs X2)
def two_sample_tests(data: SampleData, alpha: float = ALPHA) -> pd.DataFrame:
    x1, x2 = data.x1, data.x2
    n1, n2 = x1.size, x2.size

    rows = []

    # 2-sample t-test (Welch, дисперсии не равны по умолчанию)
    t_stat_w, p_w = stats.ttest_ind(x1, x2, equal_var=False, alternative="two-sided")
    rows.append(["2‑sample t‑test (Welch) H0: m1 = m2", t_stat_w, p_w, p_w < alpha])

    # 2-sample t-test (pooled, дисперсии равны)
    t_stat_p, p_p = stats.ttest_ind(x1, x2, equal_var=True, alternative="two-sided")
    rows.append(["2‑sample t‑test (pooled) H0: m1 = m2", t_stat_p, p_p, p_p < alpha])

    # 2-sample F-test (m известны): используем суммы квадратов от известных средних
    s1_known = np.sum((x1 - MU1) ** 2) / n1  # дисперсия с делением на n, df = n
    s2_known = np.sum((x2 - MU2) ** 2) / n2
    f_ratio_known = s1_known / s2_known
    dfn_k, dfd_k = n1, n2
    p_f_known = f_test_two_sided(f_ratio_known, dfn_k, dfd_k)
    rows.append(["2‑sample F‑test (m известны) H0: σ1² = σ2²", f_ratio_known, p_f_known, p_f_known < alpha])

    # 2-sample F-test (m неизвестны): отношение несмещенных дисперсий
    s1_unk = np.var(x1, ddof=1)
    s2_unk = np.var(x2, ddof=1)
    f_ratio_unk = s1_unk / s2_unk
    dfn_u, dfd_u = n1 - 1, n2 - 1
    p_f_unk = f_test_two_sided(f_ratio_unk, dfn_u, dfd_u)
    rows.append(["2‑sample F‑test (m неизвестны) H0: σ1² = σ2²", f_ratio_unk, p_f_unk, p_f_unk < alpha])

    df = pd.DataFrame(rows, columns=["Критерий", "Статистика", "p‑value", f"Решение при α={alpha}"])
    return df


# 4) Исследование распределений статистики и p‑value (под H0)
def study_statistic_distribution(N: int = 10000, seed: int = RNG_SEED) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    z_vals = np.empty(N)
    p_vals = np.empty(N)

    dfn = N1 - 1
    dfd = N2 - 1

    for i in range(N):
        x1 = rng.normal(MU1, SIGMA_1, size=N1)
        x2 = rng.normal(MU2, SIGMA_2, size=N2)
        s1 = np.var(x1, ddof=1)
        s2 = np.var(x2, ddof=1)
        z = s1 / s2
        z_vals[i] = z

        # двусторонний p‑value по F-распределению
        p = f_test_two_sided(z, dfn, dfd)

        # обрезаем из-за численных хвостов
        p_vals[i] = min(max(p, 0.0), 1.0)

    # Теоретические характеристики
    theor = {
        "Z_mean": dfd / (dfd - 2) if dfd > 2 else np.nan,
        "Z_var": (2 * (dfd ** 2) * (dfn + dfd - 2)) / (dfn * (dfd - 2) ** 2 * (dfd - 4)) if dfd > 4 else np.nan,
        "Z_std": None,
        "P_mean": 0.5,
        "P_var": 1.0 / 12.0,
        "P_std": math.sqrt(1.0 / 12.0),
    }
    theor["Z_std"] = math.sqrt(theor["Z_var"]) if theor["Z_var"] is not None and not np.isnan(theor["Z_var"]) else np.nan

    # Выборочные характеристики
    z_mean = float(np.mean(z_vals))
    z_var = float(np.var(z_vals, ddof=1))
    z_std = math.sqrt(z_var)
    p_mean = float(np.mean(p_vals))
    p_var = float(np.var(p_vals, ddof=1))
    p_std = math.sqrt(p_var)

    emp_df = pd.DataFrame(
        {
            "СВ": ["Z", "p‑value"],
            "Среднее (эмп.)": [z_mean, p_mean],
            "Дисперсия (эмп.)": [z_var, p_var],
            "СКО (эмп.)": [z_std, p_std],
        }
    )
    theor_df = pd.DataFrame(
        {
            "СВ": ["Z", "p‑value"],
            "Распределение (H0)": [f"F({dfn}, {dfd})", "Uniform(0,1)"],
            "Параметры": ["—", "—"],
            "E": [theor["Z_mean"], theor["P_mean"]],
            "Var": [theor["Z_var"], theor["P_var"]],
            "Std": [theor["Z_std"], theor["P_std"]],
        }
    )

    # Визуализация
    fig1 = plt.figure(figsize=(7, 4))
    plt.hist(z_vals, bins=60, density=True, alpha=0.6, label="эмпирическая гистограмма")
    xs = np.linspace(stats.f.ppf(0.001, dfn, dfd), stats.f.ppf(0.999, dfn, dfd), 400)
    plt.plot(xs, stats.f.pdf(xs, dfn, dfd), linewidth=2, label=f"теор. pdf F({dfn},{dfd})")
    plt.title("Гистограмма статистики Z = S1²/S2² и теор. плотность F")
    plt.xlabel("z")
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()
    fig1.canvas.manager.set_window_title("Z_hist")

    fig2 = plt.figure(figsize=(7, 4))
    plt.hist(p_vals, bins=40, density=True, alpha=0.6, label="эмпирическая гистограмма")
    xs = np.linspace(0, 1, 200)
    plt.plot(xs, np.ones_like(xs), linewidth=2, label="теор. pdf Uniform(0,1)")
    plt.title("Гистограмма p‑value и теор. плотность U(0,1)")
    plt.xlabel("p")
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()
    fig2.canvas.manager.set_window_title("pvalue_hist")

    return {"empirical": emp_df, "theoretical": theor_df}


# Отчётные таблицы печати
def print_section_1(stats_dict: Dict[str, Dict[str, float]]):
    print("\n=== 1) Выборочные характеристики ===")
    df = pd.DataFrame(stats_dict).T
    print(df.to_string(float_format=lambda v: f"{v:0.5f}"))


def print_section_2(df: pd.DataFrame):
    print("\n=== 2) Однопараметрические критерии (X1) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:0.5f}"))


def print_section_3(df: pd.DataFrame):
    print("\n=== 3) Двухвыборочные критерии (X1 vs X2) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:0.5f}"))


def print_section_4(res: Dict[str, pd.DataFrame]):
    print("\n=== 4) Теоретические характеристики (под H0) ===")
    print(res["theoretical"].to_string(index=False, float_format=lambda v: f"{v:0.5f}"))
    print("\n=== 4) Эмпирические характеристики (моделирование) ===")
    print(res["empirical"].to_string(index=False, float_format=lambda v: f"{v:0.5f}"))
    print("\nОкна с гистограммами показаны (если запуск в среде с GUI).")


# Основной сценарий
def main(alpha: float = ALPHA, N: int = 10000, seed: int = RNG_SEED):
    # номер 1: данные и выборочные характеристики
    data = generate_data(seed=seed)
    stats_dict = sample_characteristics(data)
    print_section_1(stats_dict)

    # номер 2: однопараметрические критерии
    df_one = one_sample_tests(data.x1, alpha=alpha)
    print_section_2(df_one)

    # номер 3: двухвыборочные критерии
    df_two = two_sample_tests(data, alpha=alpha)
    print_section_3(df_two)

    # номер 4: исследование распределений статистики и p-value
    res = study_statistic_distribution(N=N, seed=seed)
    print_section_4(res)

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main(alpha=0.05, N=10000, seed=2025)
