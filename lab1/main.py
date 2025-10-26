from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
from scipy import stats

# X1 ~ N(5, 2), X2 ~ N(5, 2)
MU1 = 5.0
MU2 = 5.0

SIGMA = 2.0           # истинное стандартное отклонение
SIGMA2 = SIGMA ** 2   # истинная дисперсия = 4

N1 = 100              # объем первой выборки
N2 = 100              # объем второй выборки

ALPHA = 0.05          # уровень значимости
RNG_SEED = 2025       # фиксируем сид для воспроизводимости

def two_sided_p_from_stat_cdf(stat: float, cdf_func) -> float:
    p_left = cdf_func(stat)
    p_right = 1.0 - p_left
    return 2.0 * min(p_left, p_right)

def f_test_two_sided(f_ratio: float, dfn: int, dfd: int) -> float:
    if f_ratio < 1:
        f_ratio = 1.0 / f_ratio
        dfn, dfd = dfd, dfn

    p_one_tail = stats.f.sf(f_ratio, dfn, dfd)
    # двусторонний p-value
    p_two_tail = 2.0 * min(p_one_tail, 1.0)
    return p_two_tail

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

def generate_data(seed: int = RNG_SEED) -> SampleData:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(loc=MU1, scale=SIGMA, size=N1)
    x2 = rng.normal(loc=MU2, scale=SIGMA, size=N2)
    return SampleData(x1=x1, x2=x2)

def sample_characteristics(data: SampleData) -> Dict[str, Dict[str, float]]:

    def desc(x: np.ndarray) -> Tuple[float, float, float, int]:
        d = stats.describe(x)
        mean = d.mean
        var = d.variance      # несмещенная дисперсия (ddof=1)
        std = math.sqrt(var)
        n = d.nobs
        return mean, var, std, n

    mean1, var1, std1, n1 = desc(data.x1)
    mean2, var2, std2, n2 = desc(data.x2)

    pooled_mean = float(np.mean(np.concatenate([data.x1, data.x2])))

    # классическая pooled-дисперсия (общая оценка σ², предполагая равенство дисперсий)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var)

    out = {
        "X1": {
            "mean": mean1,
            "var": var1,
            "std": std1,
            "n": n1,
        },
        "X2": {
            "mean": mean2,
            "var": var2,
            "std": std2,
            "n": n2,
        },
        "Pooled": {
            "mean": pooled_mean,
            "var": pooled_var,
            "std": pooled_std,
            "n": n1 + n2,
        },
    }
    return out

def one_sample_tests(x: np.ndarray, alpha: float = ALPHA) -> pd.DataFrame:
    n = x.size
    xbar = float(np.mean(x))
    s2 = float(np.var(x, ddof=1))  # несмещенная выборочная дисперсия

    rows = []

    #Z-test
    z_stat = (xbar - MU1) / (SIGMA / math.sqrt(n))
    z_p = two_sided_p_from_stat_cdf(z_stat, stats.norm.cdf)

    rows.append([
        f"z-test (H0: m = {MU1}, σ известна)",
        z_stat,
        z_p,
        (z_p < alpha)
    ])

    #t-test
    t_stat, t_p = stats.ttest_1samp(
        x,
        popmean=MU1,
        alternative="two-sided"
    )

    rows.append([
        f"t-test (H0: m = {MU1})",
        t_stat,
        t_p,
        (t_p < alpha)
    ])

    #χ²-test при известном m
    chi_known = np.sum((x - MU1) ** 2) / SIGMA2
    df_known = n

    p_left = stats.chi2.cdf(chi_known, df_known)
    p_right = stats.chi2.sf(chi_known, df_known)
    chi_known_p = 2.0 * min(p_left, p_right)

    rows.append([
        f"χ²-test (H0: σ² = {SIGMA2}, m известна)",
        chi_known,
        chi_known_p,
        (chi_known_p < alpha)
    ])

    #χ²-test при неизвестном m
    chi_unknown = (n - 1) * s2 / SIGMA2
    df_unknown = n - 1

    p_left = stats.chi2.cdf(chi_unknown, df_unknown)
    p_right = stats.chi2.sf(chi_unknown, df_unknown)
    chi_unknown_p = 2.0 * min(p_left, p_right)

    rows.append([
        f"χ²-test (H0: σ² = {SIGMA2}, m неизвестна)",
        chi_unknown,
        chi_unknown_p,
        (chi_unknown_p < alpha)
    ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Критерий",
            "Статистика",
            "p-value",
            f"Отклоняем H0 при α={alpha}?"
        ],
    )
    return df

def two_sample_tests(data: SampleData, alpha: float = ALPHA) -> pd.DataFrame:
    x1, x2 = data.x1, data.x2
    n1, n2 = x1.size, x2.size

    rows = []

    #t-тест
    t_stat_w, p_w = stats.ttest_ind(
        x1, x2,
        equal_var=False,
        alternative="two-sided"
    )
    rows.append([
        "2-sample t-test (Welch) H0: m1 = m2",
        t_stat_w,
        p_w,
        (p_w < alpha)
    ])

    #t-тест с равными дисперсиями
    t_stat_p, p_p = stats.ttest_ind(
        x1, x2,
        equal_var=True,
        alternative="two-sided"
    )
    rows.append([
        "2-sample t-test (pooled) H0: m1 = m2",
        t_stat_p,
        p_p,
        (p_p < alpha)
    ])

    #F-тест при известных средних
    s1_known = np.sum((x1 - MU1) ** 2) / n1
    s2_known = np.sum((x2 - MU2) ** 2) / n2
    f_ratio_known = s1_known / s2_known
    p_f_known = f_test_two_sided(f_ratio_known, n1, n2)
    rows.append([
        "2-sample F-test (m известны) H0: σ1² = σ2²",
        f_ratio_known,
        p_f_known,
        (p_f_known < alpha)
    ])

    #F-тест при неизвестных средних
    s1_unk = np.var(x1, ddof=1)
    s2_unk = np.var(x2, ddof=1)
    f_ratio_unk = s1_unk / s2_unk
    p_f_unk = f_test_two_sided(f_ratio_unk, n1 - 1, n2 - 1)
    rows.append([
        "2-sample F-test (m неизвестны) H0: σ1² = σ2²",
        f_ratio_unk,
        p_f_unk,
        (p_f_unk < alpha)
    ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Критерий",
            "Статистика",
            "p-value",
            f"Отклоняем H0 при α={alpha}?"
        ],
    )
    return df

def study_statistic_distribution(
    N: int = 10000,
    seed: int = RNG_SEED
) -> Dict[str, pd.DataFrame]:

    rng = np.random.default_rng(seed)
    z_vals = np.empty(N)
    p_vals = np.empty(N)

    dfn = N1 - 1  # степени свободы числителя
    dfd = N2 - 1  # степени свободы знаменателя

    for i in range(N):
        x1 = rng.normal(MU1, SIGMA, size=N1)
        x2 = rng.normal(MU2, SIGMA, size=N2)

        s1 = np.var(x1, ddof=1)
        s2 = np.var(x2, ddof=1)
        z = s1 / s2
        z_vals[i] = z

        # двусторонний p-value через F-распределение
        p_two = f_test_two_sided(z, dfn, dfd)

        p_vals[i] = min(max(p_two, 0.0), 1.0)

    # Теоретические характеристики F(df1=dfn, df2=dfd)
    Z_mean_theor = dfd / (dfd - 2) if dfd > 2 else np.nan
    Z_var_theor = (
        2 * (dfd ** 2) * (dfn + dfd - 2)
        / (dfn * (dfd - 2) ** 2 * (dfd - 4))
        if dfd > 4 else np.nan
    )
    Z_std_theor = math.sqrt(Z_var_theor) if not np.isnan(Z_var_theor) else np.nan

    #P-value
    P_mean_theor = 0.5
    P_var_theor = 1.0 / 12.0
    P_std_theor = math.sqrt(P_var_theor)

    # Эмпирика
    z_mean_emp = float(np.mean(z_vals))
    z_var_emp = float(np.var(z_vals, ddof=1))
    z_std_emp = math.sqrt(z_var_emp)

    p_mean_emp = float(np.mean(p_vals))
    p_var_emp = float(np.var(p_vals, ddof=1))
    p_std_emp = math.sqrt(p_var_emp)

    theoretical_df = pd.DataFrame(
        {
            "СВ": ["Z", "p-value"],
            "Распределение": [f"F({dfn}, {dfd})", "Uniform(0,1)"],
            "E": [Z_mean_theor, P_mean_theor],
            "Var": [Z_var_theor, P_var_theor],
            "Std": [Z_std_theor, P_std_theor],
        }
    )

    empirical_df = pd.DataFrame(
        {
            "СВ": ["Z", "p-value"],
            "Среднее": [z_mean_emp, p_mean_emp],
            "Дисперсия": [z_var_emp, p_var_emp],
            "СКО": [z_std_emp, p_std_emp],
        }
    )

    fig1 = plt.figure(figsize=(7, 4))
    plt.hist(z_vals, bins=60, density=True, alpha=0.6, label="эмпирическая гистограмма")
    xs = np.linspace(stats.f.ppf(0.001, dfn, dfd), stats.f.ppf(0.999, dfn, dfd), 400)
    plt.plot(xs, stats.f.pdf(xs, dfn, dfd), linewidth=2, label=f"теор. pdf F({dfn},{dfd})")
    plt.title("Гистограмма статистики Z = S1²/S2² и теоретическая плотность F")
    plt.xlabel("z")
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()

    fig2 = plt.figure(figsize=(7, 4))
    plt.hist(p_vals, bins=40, density=True, alpha=0.6, label="эмпирическая гистограмма")
    xs = np.linspace(0, 1, 200)
    plt.plot(xs, np.ones_like(xs), linewidth=2, label="теор. pdf Uniform(0,1)")
    plt.title("Гистограмма p-value и теоретическая плотность U(0,1)")
    plt.xlabel("p")
    plt.ylabel("Плотность")
    plt.legend()
    plt.tight_layout()

    return {
        "theoretical": theoretical_df,
        "empirical": empirical_df,
    }

def print_section_1(stats_dict: Dict[str, Dict[str, float]]):
    print("\n=== 1) Выборочные характеристики ===")
    df = pd.DataFrame(stats_dict).T
    print(df.to_string(float_format=lambda v: f"{v:0.5f}"))


def print_section_2(df: pd.DataFrame):
    print("\n=== 2) Одновыборочные критерии (X1) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:0.5f}"))


def print_section_3(df: pd.DataFrame):
    print("\n=== 3) Двухвыборочные критерии (X1 vs X2) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:0.5f}"))


def print_section_4(res: Dict[str, pd.DataFrame]):
    print("\n=== 4) Теоретические характеристики (под H0) ===")
    print(res["theoretical"].to_string(index=False, float_format=lambda v: f"{v:0.5f}"))
    print("\n=== 4) Эмпирические характеристики (моделирование) ===")
    print(res["empirical"].to_string(index=False, float_format=lambda v: f"{v:0.5f}"))
    print("\n(См. также гистограммы Z и p-value.)")

def main(alpha: float = ALPHA, N: int = 10000, seed: int = RNG_SEED):
    # 1) данные и выборочные характеристики
    data = generate_data(seed=seed)
    stats_dict = sample_characteristics(data)
    print_section_1(stats_dict)

    # 2) одновыборочные критерии
    df_one = one_sample_tests(data.x1, alpha=alpha)
    print_section_2(df_one)

    # 3) двухвыборочные критерии
    df_two = two_sample_tests(data, alpha=alpha)
    print_section_3(df_two)

    # 4) исследование распределений статистики и p-value
    res = study_statistic_distribution(N=N, seed=seed)
    print_section_4(res)

    # показать графики
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main(alpha=0.05, N=10000, seed=2025)
