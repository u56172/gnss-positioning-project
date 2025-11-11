from mpmath import mp, matrix, det, inverse, diag, sqrt
import math
from pyproj import Transformer

# Ustawianie precyzji
mp.dps = 50

# Dane ośmiu satelitów
# E25 E33 E05 E24 E26 E10 E12 E03

# Współrzędne z pliku COD0MGXFIN_20232390000_01D_05M_ORB.SP3
X_P1 = mp.mpf('9165.422218') * 1000
Y_P1 = mp.mpf('-21173.670524') * 1000
Z_P1 = mp.mpf('18553.809199') * 1000

X_P2 = mp.mpf('6015.175637') * 1000
Y_P2 = mp.mpf('17739.850894') * 1000
Z_P2 = mp.mpf('22919.827867') * 1000

X_P3 = mp.mpf('19940.333610') * 1000
Y_P3 = mp.mpf('-21357.106773') * 1000
Z_P3 = mp.mpf('4748.811857') * 1000

X_P4 = mp.mpf('17004.601067') * 1000
Y_P4 = mp.mpf('-11.655858') * 1000
Z_P4 = mp.mpf('24223.076276') * 1000

X_P5 = mp.mpf('-14133.017403') * 1000
Y_P5 = mp.mpf('11212.261574') * 1000
Z_P5 = mp.mpf('23462.990565') * 1000

X_P6 = mp.mpf('28103.284609') * 1000
Y_P6 = mp.mpf('9279.665249') * 1000
Z_P6 = mp.mpf('-875.769003') * 1000

X_P7 = mp.mpf('24397.935380') * 1000
Y_P7 = mp.mpf('14466.511132') * 1000
Z_P7 = mp.mpf('8444.458528') * 1000

X_P8 = mp.mpf('3355.272678') * 1000
Y_P8 = mp.mpf('-21413.041259') * 1000
Z_P8 = mp.mpf('20159.586426') * 1000

satelity = {
    "PE25": {"X": X_P1, "Y": Y_P1, "Z": Z_P1},
    "PE33": {"X": X_P2, "Y": Y_P2, "Z": Z_P2},
    "PE05": {"X": X_P3, "Y": Y_P3, "Z": Z_P3},
    "PE24": {"X": X_P4, "Y": Y_P4, "Z": Z_P4},
    "PE26": {"X": X_P5, "Y": Y_P5, "Z": Z_P5},
    "PE10": {"X": X_P6, "Y": Y_P6, "Z": Z_P6},
    "PE12": {"X": X_P7, "Y": Y_P7, "Z": Z_P7},
    "PE03": {"X": X_P8, "Y": Y_P8, "Z": Z_P8}
}

# Pseudoodległości z pliku RINEX PTBB00DEU_R_20232390000_01D_30S_MO.rnx (pomiar C1C)
PR1 = 26273493.271
PR2 = 24800225.406
PR3 = 27311202.472
PR4 = 23374936.036
PR5 = 27750716.795
PR6 = 26563786.602
PR7 = 25170359.853
PR8 = 26824821.173

PR = [PR1, PR2, PR3, PR4, PR5, PR6, PR7, PR8]

# Poprawki do czasu zegara z pliku COD0MGXFIN_20232390000_01D_30S_CLK.CLK
dt1 = -0.858542639667E-06
dt2 = -0.112310425685E-05
dt3 = 0.127153868909E-04
dt4 = -0.287827488956E-03
dt5 = 0.204505790865E-03
dt6 = -0.559104862704E-03
dt7 = -0.673569919106E-03
dt8 = -0.498754272602E-04

dt = [dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8]

# Prędkość światła [w metrach]
c = 299792458

# Współrzędne katalogowe odbiornika GNSS
w_kat = {
    'Xr': 3844059.677,
    'Yr': 709661.621,
    'Zr': 5023129.738
}

# Skorygowane PR poprzez dodanie c*dt poprawki czasu w celu późnejszego zamieszczenia w macierzy punkt 4
dtc = [dt[i] * c for i in range(8)]
PR_poprawka = [PR[i] + dtc[i] for i in range(8)]

# Macierz B
B = matrix([
    [X_P1, Y_P1, Z_P1, PR_poprawka[0]],
    [X_P2, Y_P2, Z_P2, PR_poprawka[1]],
    [X_P3, Y_P3, Z_P3, PR_poprawka[2]],
    [X_P4, Y_P4, Z_P4, PR_poprawka[3]],
    [X_P5, Y_P5, Z_P5, PR_poprawka[4]],
    [X_P6, Y_P6, Z_P6, PR_poprawka[5]],
    [X_P7, Y_P7, Z_P7, PR_poprawka[6]],
    [X_P8, Y_P8, Z_P8, PR_poprawka[7]]
])


def iloczyn_lorentza(a, b):
    M = mp.diag([1, 1, 1, -1])
    a = matrix(a)
    b = matrix(b)
    return (a.T @ M @ b)[0, 0]


# Wprowadzony zostaje również wektor a:
def oblicz_wektor_a(satelity: dict, PR: list, dt: list):
    """Zwraca słownik dla wszystkich satelitów."""
    wynik = {}
    for i, (key, sat) in enumerate(satelity.items()):
        temp = matrix([sat['X'], sat['Y'], sat['Z'], PR[i] + c * dt[i]])
        a_i = 0.5 * iloczyn_lorentza(temp, temp)
        wynik[f'a{i}'] = a_i
    return wynik


def oblicz_wspolrzedne_odbiornika(satelity: dict, PR: list, dt: list, B: matrix):
    """Zwraca dwa możliwe rozwiązania współrzędnych odbiornika: (Xr1, Yr1, Zr1, cdtr1) oraz (Xr2, Yr2, Zr2, cdtr2)"""

    M = mp.diag([1, 1, 1, -1])
    I = matrix([[1], [1], [1], [1], [1], [1], [1], [1]])
    wektory_a = oblicz_wektor_a(satelity, PR, dt)
    a = matrix([[v] for v in wektory_a.values()])

    BT = B.T
    BT_B = BT @ B
    BT_B_inv = inverse(BT_B)

    a_quad = iloczyn_lorentza(BT_B_inv @ (BT @ I), BT_B_inv @ (BT @ I))
    b_quad = 2 * (iloczyn_lorentza(BT_B_inv @ (BT @ I), BT_B_inv @ (BT @ a)) - 1)
    c_quad = iloczyn_lorentza(BT_B_inv @ (BT @ a), BT_B_inv @ (BT @ a))

    nowa_delta = b_quad ** 2 - (4 * a_quad * c_quad)
    nowadelta_sqrt = sqrt(nowa_delta)

    lambda1 = ((-1) * b_quad + nowadelta_sqrt) / (2 * a_quad)
    lambda2 = ((-1) * b_quad - nowadelta_sqrt) / (2 * a_quad)
    # [ 𝑟 𝑐𝑑𝑡𝑟 ]T^-1=𝑀𝐵−1(𝛬𝐼+𝑎)
    rcdtr1 = M @ (BT_B_inv @ (BT @ (lambda1 * I + a)))
    rcdtr2 = M @ (BT_B_inv @ (BT @ (lambda2 * I + a)))

    return (
        rcdtr1[0], rcdtr1[1], rcdtr1[2], rcdtr1[3],
        rcdtr2[0], rcdtr2[1], rcdtr2[2], rcdtr2[3]
    )


Xr1, Yr1, Zr1, cdtr1, Xr2, Yr2, Zr2, cdtr2 = oblicz_wspolrzedne_odbiornika(satelity, PR, dt, B)
# Wybieram Xr2 które znajduje się bliżej Ziemi
'''
print(Xr2)
print(Yr2)
print(Zr2)
'''

# Różnice katalogowe - wyliczone
roznice_kat_wyl = {
    'dx': (w_kat['Xr'] - Xr2),
    'dy': (w_kat['Yr'] - Yr2),
    'dz': (w_kat['Zr'] - Zr2)
}

# Przeliczenie współrzędnych na elipsoidalne
transformacja = Transformer.from_crs("epsg:9988", "epsg:9989")
x, y, z = Xr2, Yr2, Zr2
szerokosc, dlugosc, H = transformacja.transform(x, y, z)

# Model Sasstamoinen'a

P = 1013.25 * (1 - 2.2557 * 10E-5 * H) ** 5.2568
szerokosc_radiany = math.radians(szerokosc)
dlugosc_radiany = math.radians(dlugosc)
ZHD = (0.0022768 * P) / (1 - 0.00266 * math.cos(2 * szerokosc_radiany) - 2.8 * 10E-7 * H)
Pw = 7
ZWD = 0.0122 + 0.00943 * Pw


# Wyliczanie elewacji
def oblicz_elev(Xs, Ys, Zs, Xr, Yr, Zr, szerokosc, dlugosc):
    e_ = (-math.sin(dlugosc), math.cos(dlugosc), 0)
    n_ = (-math.cos(dlugosc) * math.sin(szerokosc), -math.sin(dlugosc) * math.sin(szerokosc), math.cos(szerokosc))
    u_ = (math.cos(dlugosc) * math.cos(szerokosc), math.sin(dlugosc) * math.cos(szerokosc), math.sin(szerokosc))

    dx, dy, dz = (Xs - Xr), (Ys - Yr), (Zs - Zr)

    wektor_normalny = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    rho = (dx / wektor_normalny, dy / wektor_normalny, dz / wektor_normalny)

    e_iloczyn_skalarny = sum(rho[i] * e_[i] for i in range(3))
    n_iloczyn_skalarny = sum(rho[i] * n_[i] for i in range(3))
    u_iloczyn_skalarny = sum(rho[i] * u_[i] for i in range(3))

    E = math.asin(u_iloczyn_skalarny)
    A = math.atan(e_iloczyn_skalarny / n_iloczyn_skalarny)

    E_stopnie = math.degrees(E)
    A_stopnie = math.degrees(A)

    if A_stopnie < 0:
        A_stopnie += 360

    return E_stopnie, A_stopnie


# Funkcja odwzorowująca Black and Eisner

def black_and_eisner(ele, ZHD, ZWD):
    ele_radiany = math.radians(ele)
    mfele = 1.001 / math.sqrt(0.002001 + (math.sin(ele_radiany)) ** 2)
    ZTD = ZWD + ZHD
    STD = mfele * ZTD
    return STD, mfele


std_lista = []
for nazwa, sat in satelity.items():
    ele, azym = oblicz_elev(sat["X"], sat["Y"], sat["Z"], Xr2, Yr2, Zr2, szerokosc_radiany, dlugosc_radiany)
    std, mfele = black_and_eisner(ele, ZHD, ZWD)
    std_lista.append(std)
    # print(f"{nazwa}: Elewacja = {ele:.9f}, Azymut = {azym:.9f}, STD = {std:.3f}, mfele = {mfele:.3f}")

'''
print(std_lista)
print(ZHD)
print(ZWD)
print(ZWD+ZHD)
'''

# Zadeklarowanie PRE5 czyli C5X z rinex

C5X_1 = 26273495.192
C5X_2 = 24800225.328
C5X_3 = 27311204.955
C5X_4 = 23374935.909
C5X_5 = 27750717.695
C5X_6 = 26563789.568
C5X_7 = 25170359.027
C5X_8 = 26824822.649

C5X = [C5X_1, C5X_2, C5X_3, C5X_4, C5X_5, C5X_6, C5X_7, C5X_8]

E1 = 1575.42
E5 = 1176.42

PR_IF_lista = [(E1 ** 2) / (E1 ** 2 - E5 ** 2) * PR[i] - (E5 ** 2) / (E1 ** 2 - E5 ** 2) * C5X[i]
               for i in range(8)
               ]


def iteracyjne_dopasowanie_pozycji(Xr2, Yr2, Zr2, satelity, PR_IF_lista, std_lista, dtc, iteracje):
    """
    Wykonuje iteracyjne dopasowanie pozycji odbiornika GNSS.
    Zwraca zaktualizowane współrzędne (Xr, Yr, Zr).
    """
    for i in range(4):
        # Odległości geometryczne do każdego z satelitów
        ro_r_lista = [
            math.sqrt((sat["X"] - Xr2) ** 2 + (sat["Y"] - Yr2) ** 2 + (sat["Z"] - Zr2) ** 2)
            for sat in satelity.values()
        ]
        # Macierz A
        A_ = matrix([
            [(Xr2 - satelity[key]["X"]) / ro_r_lista[i],
             (Yr2 - satelity[key]["Y"]) / ro_r_lista[i],
             (Zr2 - satelity[key]["Z"]) / ro_r_lista[i],
             1]
            for j, key in enumerate(satelity.keys())
        ])
        # Wektor L
        L = matrix([
            PR_IF_lista[i] - ro_r_lista[i] - std_lista[i] + dtc[i]
            for j in range(8)
        ])
        # Korekty pozycji
        AT = A_.T
        covA = inverse(AT @ A_) @ AT @ L
        dx, dy, dz, c_dt_r = covA[0, 0], covA[1, 0], covA[2, 0], covA[3, 0]
        # Aktualizacja współrzędnych
        Xr2 += dx
        Yr2 += dy
        Zr2 += dz

        # print(f"Iteracja {i + 1}:")
        # print(f"dx = {float(dx)}, dy = {float(dy)}, dz = {float(dz)}")
        # print(f"Zaktualizowane wspolrzedne: Xr2 = {float(Xr2):.3f}, Yr2 = {float(Yr2):.3f}, Zr2 = {float(Zr2):.3f}")

        return Xr2, Yr2, Zr2


Xr2, Yr2, Zr2 = iteracyjne_dopasowanie_pozycji(Xr2, Yr2, Zr2, satelity, PR_IF_lista, std_lista, dtc, 4)
roz_dx = w_kat['Xr'] - Xr2
roz_dy = w_kat['Yr'] - Yr2
roz_dz = w_kat['Zr'] - Zr2

print(f"ΔX={float(roz_dx):.3f}")
print(f"ΔY={float(roz_dy):.3f}")
print(f"ΔZ={float(roz_dz):.3f}")

