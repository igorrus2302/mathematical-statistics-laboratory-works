import numpy as np
from scipy.stats import (
    pearsonr,
    spearmanr,
    kendalltau,
    chi2_contingency,
    rankdata,
)
import matplotlib.pyplot as plt


def print_title(title: str):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def print_table(headers, rows, float_prec=4):
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.{float_prec}f}"
        return str(x)

    str_rows = [[fmt(cell) for cell in row] for row in rows]
    str_headers = [str(h) for h in headers]

    # ширина колонок
    cols = list(zip(str_headers, *str_rows))
    col_widths = [max(len(cell) for cell in col) for col in cols]

    def print_row(row):
        parts = []
        for w, cell in zip(col_widths, row):
            parts.append(cell.ljust(w))
        print(" | ".join(parts))

    sep = "-+-".join("-" * w for w in col_widths)

    print_row(str_headers)
    print(sep)
    for r in str_rows:
        print_row(r)


def decision(p, alpha=0.05):
    return "отвергаем H0" if p < alpha else "не отвергаем H0"


SEED = 5
rng = np.random.default_rng(SEED)

n = 100
mu_x, sigma_x = 5, 2
mu_y, sigma_y = 5, 2

alpha = 0.05

X = rng.normal(loc=mu_x, scale=sigma_x, size=n)
Y = rng.normal(loc=mu_y, scale=sigma_y, size=n)

print_title("Выборочные характеристики")

mean_X = np.mean(X)
mean_Y = np.mean(Y)
var_X = np.var(X, ddof=1)
var_Y = np.var(Y, ddof=1)

r_xy, p_pearson = pearsonr(X, Y)
rho_s, p_spearman = spearmanr(X, Y)
tau_k, p_kendall = kendalltau(X, Y)

headers = ["СВ", "x̄", "s²", "r (Пирсон)", "ρ_S (Спирмен)", "τ (Кендалл)"]
rows = [
    ["X", mean_X, var_X, "-", "-", "-"],
    ["Y", mean_Y, var_Y, "-", "-", "-"],
    ["(X,Y)", "-", "-", r_xy, rho_s, tau_k],
]
print_table(headers, rows)

print_title("Проверка значимости коэффициентов корреляции")

headers = ["Коэффициент", "Оценка", "p-value", f"Решение при α={alpha}"]
rows = [
    ["Пирсон",  r_xy,   p_pearson,   decision(p_pearson, alpha)],
    ["Спирмен", rho_s,  p_spearman,  decision(p_spearman, alpha)],
    ["Кендалл", tau_k,  p_kendall,   decision(p_kendall, alpha)],
]
print_table(headers, rows)

plt.figure(figsize=(6, 5))
plt.scatter(X, Y, edgecolors="k", alpha=0.7)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Диаграмма рассеяния X и Y")
plt.grid(True)
plt.tight_layout()
plt.show()

print_title("Проверка независимости методом таблиц сопряженности")

bins = 5
H, xedges, yedges = np.histogram2d(X, Y, bins=bins)

print("Интервалы по X:")
print(*[f"[{xedges[i]:.2f}; {xedges[i+1]:.2f})" for i in range(len(xedges)-1)], sep="\n")
print("\nИнтервалы по Y:")
print(*[f"[{yedges[i]:.2f}; {yedges[i+1]:.2f})" for i in range(len(yedges)-1)], sep="\n")

headers = ["X\\Y"] + [f"[{yedges[j]:.2f}; {yedges[j+1]:.2f})" for j in range(bins)]
rows = []
for i in range(bins):
    row = [f"[{xedges[i]:.2f}; {xedges[i+1]:.2f})"] + list(H[i, :])
    rows.append(row)

print("\nЭмпирическая таблица сопряженности:")
print_table(headers, rows, float_prec=0)

chi2, p_chi2, dof, expected = chi2_contingency(H)

print(f"\nСтатистика χ² = {chi2:.4f}, df = {dof}, p-value = {p_chi2:.4f}")
print(f"Решение при α={alpha}: {decision(p_chi2, alpha)}")

rows_exp = []
for i in range(bins):
    row = [f"[{xedges[i]:.2f}; {xedges[i+1]:.2f})"] + list(expected[i, :])
    rows_exp.append(row)

print("\nТеоретическая таблица сопряженности:")
print_table(headers, rows_exp, float_prec=2)

lambdas = np.linspace(0, 1, 51)

r_XU, rho_XU, tau_XU = [], [], []
r_XV, rho_XV, tau_XV = [], [], []

for lam in lambdas:
    U = X + (1 - lam) * Y
    V = X**3 + (1 - lam) * (Y**3)

    r1,  _ = pearsonr(X, U)
    rs1, _ = spearmanr(X, U)
    tk1, _ = kendalltau(X, U)

    r2,  _ = pearsonr(X, V)
    rs2, _ = spearmanr(X, V)
    tk2, _ = kendalltau(X, V)

    r_XU.append(r1);   rho_XU.append(rs1);   tau_XU.append(tk1)
    r_XV.append(r2);   rho_XV.append(rs2);   tau_XV.append(tk2)

r_XU   = np.array(r_XU)
rho_XU = np.array(rho_XU)
tau_XU = np.array(tau_XU)

r_XV   = np.array(r_XV)
rho_XV = np.array(rho_XV)
tau_XV = np.array(tau_XV)

plt.figure(figsize=(7, 5))
plt.plot(lambdas, r_XU,   label="r_XU(λ)  (Пирсон)")
plt.plot(lambdas, rho_XU, label="ρ_XU(λ) (Спирмен)")
plt.plot(lambdas, tau_XU, label="τ_XU(λ) (Кендалл)")
plt.xlabel("λ")
plt.ylabel("значение коэффициента")
plt.title("Графики зависимостей r_XU(λ), ρ_XU(λ), τ_XU(λ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(lambdas, r_XV,   label="r_XV(λ)  (Пирсон)")
plt.plot(lambdas, rho_XV, label="ρ_XV(λ) (Спирмен)")
plt.plot(lambdas, tau_XV, label="τ_XV(λ) (Кендалл)")
plt.xlabel("λ")
plt.ylabel("значение коэффициента")
plt.title("Графики зависимостей r_XV(λ), ρ_XV(λ), τ_XV(λ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def scatter_and_rank_plots(lam):
    V = X**3 + (1 - lam) * (Y**3)

    plt.figure(figsize=(6, 5))
    plt.scatter(X, V, edgecolors="k", alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("V")
    plt.title(f"Диаграмма рассеяния X и V при λ = {lam}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    Rx = rankdata(X)
    Rv = rankdata(V)

    plt.figure(figsize=(6, 5))
    plt.scatter(Rx, Rv, edgecolors="k", alpha=0.7)
    plt.xlabel("rang X")
    plt.ylabel("rang V")
    plt.title(f"Диаграмма рассеяния рангов X и V при λ = {lam}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


scatter_and_rank_plots(0.0)
scatter_and_rank_plots(1.0)
