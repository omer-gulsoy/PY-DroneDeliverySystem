import random
import datetime
import time
from typing import List, Tuple
import math
import heapq
import matplotlib.pyplot as plt

# -------------------------
# Veri Modelleri
# -------------------------

class Ucak:
    def __init__(self, idx: int, maks_agirlik: float, batarya: int, hiz: float, baslangic: Tuple[float, float]):
        self.idx = idx
        self.maks_agirlik = maks_agirlik
        self.batarya = batarya
        self.hiz = hiz
        self.baslangic = baslangic
        self.anlik_konum = baslangic
        self.kalan_batarya = batarya
        self.toplam_mesafe = 0
        self.toplam_enerji = 0
        self.tamamlanan_teslimat = 0

    def __str__(self):
        return f"Ucak {self.idx} (Max Ağırlık: {self.maks_agirlik}kg, Batarya: {self.batarya}mAh, Hız: {self.hiz}m/s)"


class TeslimatNoktasi:
    def __init__(self, idx: int, konum: Tuple[float, float], agirlik: float, oncelik: int,
                 zaman_araligi: Tuple[datetime.time, datetime.time]):
        self.idx = idx
        self.konum = konum
        self.agirlik = agirlik
        self.oncelik = oncelik
        self.zaman_araligi = zaman_araligi
        self.teslim_edildi = False

    def __eq__(self, diger):
        return isinstance(diger, TeslimatNoktasi) and self.idx == diger.idx

    def __hash__(self):
        return hash(self.idx)

    def __str__(self):
        return f"Teslimat {self.idx} (Konum: {self.konum}, Ağırlık: {self.agirlik}kg, Öncelik: {self.oncelik})"


class UcusYasakBolgesi:
    def __init__(self, idx: int, koordinatlar: List[Tuple[float, float]],
                 aktif_zaman: Tuple[datetime.time, datetime.time]):
        self.idx = idx
        self.koordinatlar = koordinatlar
        self.aktif_zaman = aktif_zaman

    def __str__(self):
        return f"Yasak Bölge {self.idx} (Köşe sayısı: {len(self.koordinatlar)})"


# -------------------------
# Yardımcı Fonksiyonlar
# -------------------------

def oklid_mesafe(n1: Tuple[float, float], n2: Tuple[float, float]) -> float:
    return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)


def nokta_polison_ici(nokta: Tuple[float, float], poligon: List[Tuple[float, float]]) -> bool:
    x, y = nokta
    n = len(poligon)
    inside = False
    p1x, p1y = poligon[0]
    for i in range(n + 1):
        p2x, p2y = poligon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def nokta_yasak_bolge_ici(nokta: Tuple[float, float], zaman: datetime.time,
                          yasak_bolgeler: List[UcusYasakBolgesi]) -> bool:
    for bolge in yasak_bolgeler:
        if bolge.aktif_zaman[0] <= zaman <= bolge.aktif_zaman[1]:
            if nokta_polison_ici(nokta, bolge.koordinatlar):
                return True
    return False


def cizgi_parcasi_kesisiyor_mu(p1: Tuple[float, float], p2: Tuple[float, float],
                               e1: Tuple[float, float], e2: Tuple[float, float]) -> bool:
    def oryantasyon(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def seg_on_segment(a, b, c):
        return (b[0] <= max(a[0], c[0]) and b[0] >= min(a[0], c[0]) and
                b[1] <= max(a[1], c[1]) and b[1] >= min(a[1], c[1]))

    o1 = oryantasyon(p1, p2, e1)
    o2 = oryantasyon(p1, p2, e2)
    o3 = oryantasyon(e1, e2, p1)
    o4 = oryantasyon(e1, e2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and seg_on_segment(p1, e1, p2): return True
    if o2 == 0 and seg_on_segment(p1, e2, p2): return True
    if o3 == 0 and seg_on_segment(e1, p1, e2): return True
    if o4 == 0 and seg_on_segment(e1, p2, e2): return True

    return False


def cizgi_poligon_kesisiyor_mu(p1: Tuple[float, float], p2: Tuple[float, float],
                               poligon: List[Tuple[float, float]]) -> bool:
    n = len(poligon)
    for i in range(n):
        if cizgi_parcasi_kesisiyor_mu(p1, p2, poligon[i], poligon[(i + 1) % n]):
            return True
    return False


def yol_yasak_bolge_iceriyor_mu(p1: Tuple[float, float], p2: Tuple[float, float], zaman: datetime.time,
                                yasak_bolgeler: List[UcusYasakBolgesi]) -> bool:
    if nokta_yasak_bolge_ici(p1, zaman, yasak_bolgeler) or nokta_yasak_bolge_ici(p2, zaman, yasak_bolgeler):
        return True
    for bolge in yasak_bolgeler:
        if bolge.aktif_zaman[0] <= zaman <= bolge.aktif_zaman[1]:
            if cizgi_poligon_kesisiyor_mu(p1, p2, bolge.koordinatlar):
                return True
    return False


def kenar_maliyeti(h1: Tuple[float, float], h2: Tuple[float, float],
                   oncelik: int, hiz: float) -> float:
    uzaklik = oklid_mesafe(h1, h2)
    return uzaklik * hiz + (oncelik * 100)


def enerji_hesapla(uzaklik: float, agirlik: float, maks_agirlik: float) -> float:
    return uzaklik * (1 + agirlik / maks_agirlik)


# -------------------------
# A* Algoritması
# -------------------------

class Dugum:
    def __init__(self, pos: Tuple[float, float], g=0, h=0, parent=None):
        self.pos = pos
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return isinstance(other, Dugum) and self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)


def a_yildiz(baslangic: Tuple[float, float], hedef: Tuple[float, float],
             yasak_bolgeler: List[UcusYasakBolgesi], zaman: datetime.time,
             oncelik: int = 1, hiz: float = 10.0,
             harita_boyutu=100, adim=2.0) -> List[Tuple[float, float]]:

    acik_kume = []
    kapali_kume = set()

    bas_node = Dugum(baslangic)
    bas_node.h = oklid_mesafe(baslangic, hedef)
    bas_node.f = bas_node.g + bas_node.h

    heapq.heappush(acik_kume, (bas_node.f, id(bas_node), bas_node))
    dugum_dict = {baslangic: bas_node}

    max_dongu = 1000
    sayac = 0

    while acik_kume and sayac < max_dongu:
        sayac += 1
        _, _, mevcut = heapq.heappop(acik_kume)

        if oklid_mesafe(mevcut.pos, hedef) < adim:
            yol = []
            while mevcut:
                yol.append(mevcut.pos)
                mevcut = mevcut.parent
            return yol[::-1]

        kapali_kume.add(mevcut.pos)

        hareket_yonleri = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in hareket_yonleri:
            komsu = (mevcut.pos[0] + dx * adim, mevcut.pos[1] + dy * adim)

            if not (0 <= komsu[0] <= harita_boyutu and 0 <= komsu[1] <= harita_boyutu):
                continue

            if yol_yasak_bolge_iceriyor_mu(mevcut.pos, komsu, zaman, yasak_bolgeler):
                continue

            if komsu in kapali_kume:
                continue

            kenar_m = kenar_maliyeti(mevcut.pos, komsu, oncelik, hiz)
            gecici_g = mevcut.g + kenar_m

            if komsu in dugum_dict:
                komsu_dugum = dugum_dict[komsu]
                if gecici_g >= komsu_dugum.g:
                    continue
            else:
                komsu_dugum = Dugum(komsu)
                dugum_dict[komsu] = komsu_dugum

            komsu_dugum.parent = mevcut
            komsu_dugum.g = gecici_g

            base_h = oklid_mesafe(komsu, hedef)
            ceza = 0
            if yol_yasak_bolge_iceriyor_mu(komsu, hedef, zaman, yasak_bolgeler):
                ceza = 1000

            komsu_dugum.h = base_h + ceza
            komsu_dugum.f = komsu_dugum.g + komsu_dugum.h

            if komsu not in [n.pos for _, _, n in acik_kume]:
                heapq.heappush(acik_kume, (komsu_dugum.f, id(komsu_dugum), komsu_dugum))

    if not yol_yasak_bolge_iceriyor_mu(baslangic, hedef, zaman, yasak_bolgeler):
        return [baslangic, hedef]

    return []


# -------------------------
# Rota ve Genetik Algoritma
# -------------------------

class Rota:
    def __init__(self, ucak: Ucak, teslimatlar: List[TeslimatNoktasi] = None):
        self.ucak = ucak
        self.teslimatlar = teslimatlar if teslimatlar else []
        self.yol = []
        self.noktalar = []
        self.maliyet = float('inf')
        self.enerji_tuketimi = 0
        self.toplam_mesafe = 0
        self.toplam_oncelik = 0
        self.gecerli = True

    def teslimat_ekle(self, teslimat: TeslimatNoktasi) -> bool:
        agirlik_toplam = sum(t.agirlik for t in self.teslimatlar)
        if agirlik_toplam + teslimat.agirlik > self.ucak.maks_agirlik:
            return False
        self.teslimatlar.append(teslimat)
        return True

    def yol_hesapla(self, yasak_bolgeler: List[UcusYasakBolgesi], zaman: datetime.time):
        if not self.teslimatlar:
            self.yol = []
            self.noktalar = []
            self.maliyet = 0
            self.enerji_tuketimi = 0
            self.toplam_mesafe = 0
            self.toplam_oncelik = 0
            self.gecerli = True
            return

        self.teslimatlar.sort(key=lambda x: x.oncelik, reverse=True)
        self.noktalar = [self.ucak.baslangic] + [t.konum for t in self.teslimatlar] + [self.ucak.baslangic]

        tam_yol = []
        toplam_mesafe = 0
        toplam_maliyet = 0
        toplam_enerji = 0
        toplam_oncelik = sum(t.oncelik for t in self.teslimatlar)

        for i in range(len(self.noktalar) - 1):
            oncelik = 1
            if 0 < i <= len(self.teslimatlar):
                oncelik = self.teslimatlar[i - 1].oncelik

            parca = a_yildiz(
                self.noktalar[i],
                self.noktalar[i + 1],
                yasak_bolgeler,
                zaman,
                oncelik,
                self.ucak.hiz
            )

            if not parca:
                self.yol = []
                self.maliyet = float('inf')
                self.enerji_tuketimi = float('inf')
                self.toplam_mesafe = 0
                self.toplam_oncelik = 0
                self.gecerli = False
                return

            if i > 0:
                parca = parca[1:]

            tam_yol.extend(parca)

            mesafe_parca = 0
            for j in range(len(parca) - 1):
                mesafe_parca += oklid_mesafe(parca[j], parca[j + 1])
            toplam_mesafe += mesafe_parca

            maliyet_parca = kenar_maliyeti(self.noktalar[i], self.noktalar[i + 1], oncelik, self.ucak.hiz)
            toplam_maliyet += maliyet_parca

            agirlik_parca = 0
            if i < len(self.teslimatlar):
                agirlik_parca = self.teslimatlar[i].agirlik

            enerji_parca = enerji_hesapla(mesafe_parca, agirlik_parca, self.ucak.maks_agirlik)
            toplam_enerji += enerji_parca

        self.yol = tam_yol
        self.maliyet = toplam_maliyet
        self.enerji_tuketimi = toplam_enerji
        self.toplam_mesafe = toplam_mesafe
        self.toplam_oncelik = toplam_oncelik
        self.gecerli = True

    def __str__(self):
        return (f"Ucak {self.ucak.idx} Rota: {len(self.teslimatlar)} teslimat, "
                f"Maliyet: {self.maliyet:.2f}, Enerji: {self.enerji_tuketimi:.2f}")


def kisitlar_kontrol(route: Rota, yasak_bolgeler: List[UcusYasakBolgesi], zaman: datetime.time) -> bool:
    toplam_agirlik = sum(t.agirlik for t in route.teslimatlar)
    if toplam_agirlik > route.ucak.maks_agirlik:
        return False

    simdi = datetime.datetime.now().time()
    for t in route.teslimatlar:
        if not (t.zaman_araligi[0] <= simdi <= t.zaman_araligi[1]):
            return False

    route.yol_hesapla(yasak_bolgeler, zaman)
    if not route.gecerli or route.maliyet == float('inf'):
        return False

    if route.enerji_tuketimi > route.ucak.batarya / 1000:
        return False

    return True


def uygunluk_degeri(rotalar: List[Rota], yasak_bolgeler: List[UcusYasakBolgesi], zaman: datetime.time) -> float:
    toplam_teslimat = 0
    toplam_enerji = 0
    ihlal_sayisi = 0

    for r in rotalar:
        if not kisitlar_kontrol(r, yasak_bolgeler, zaman):
            ihlal_sayisi += 1
            continue
        toplam_teslimat += len(r.teslimatlar)
        toplam_enerji += r.enerji_tuketimi

    return (toplam_teslimat * 100) - (toplam_enerji * 0.5) - (ihlal_sayisi * 2000)


def caprazlama(ebeveyn1: List[Rota], ebeveyn2: List[Rota], ucaklar: List[Ucak]) -> List[Rota]:
    cocuk = []
    for i in range(len(ucaklar)):
        t1 = ebeveyn1[i].teslimatlar.copy() if i < len(ebeveyn1) else []
        t2 = ebeveyn2[i].teslimatlar.copy() if i < len(ebeveyn2) else []

        kesme = random.randint(0, len(t1))

        cocuk_teslimatlar = t1[:kesme]

        for t in t2:
            if t not in cocuk_teslimatlar:
                cocuk_teslimatlar.append(t)

        cocuk.append(Rota(ucaklar[i], cocuk_teslimatlar))

    return cocuk


def mutasyon(rotalar: List[Rota], tum_teslimatlar: List[TeslimatNoktasi], oran=0.1):
    atanmis = set()
    for r in rotalar:
        for t in r.teslimatlar:
            atanmis.add(t)

    atanmayanlar = [t for t in tum_teslimatlar if t not in atanmis]

    for r in rotalar:
        if random.random() < oran and len(r.teslimatlar) > 1:
            i1, i2 = random.sample(range(len(r.teslimatlar)), 2)
            r.teslimatlar[i1], r.teslimatlar[i2] = r.teslimatlar[i2], r.teslimatlar[i1]

        if random.random() < oran / 2 and rotalar and r.teslimatlar:
            diger = random.choice([x for x in rotalar if x != r])
            if r.teslimatlar:
                t = random.choice(r.teslimatlar)
                r.teslimatlar.remove(t)
                diger.teslimatlar.append(t)

        if random.random() < oran / 3 and atanmayanlar:
            t = random.choice(atanmayanlar)
            if r.teslimat_ekle(t):
                atanmayanlar.remove(t)

        if random.random() < oran / 4 and r.teslimatlar:
            t = random.choice(r.teslimatlar)
            r.teslimatlar.remove(t)
            atanmayanlar.append(t)

    random.shuffle(atanmayanlar)
    for t in atanmayanlar:
        random.shuffle(rotalar)
        for r in rotalar:
            if r.teslimat_ekle(t):
                break


def genetik_optim(dronlar: List[Ucak], teslimatlar: List[TeslimatNoktasi], yasak_bolgeler: List[UcusYasakBolgesi],
                  nesil_sayisi=50, pop_boyutu=30):
    simdi = datetime.datetime.now().time()
    baslangic = time.time()

    populasyon = []
    for _ in range(pop_boyutu):
        random.shuffle(teslimatlar)
        rotalar = []
        kalan = teslimatlar.copy()

        for d in dronlar:
            r = Rota(d)
            rotalar.append(r)
            sec_say = random.randint(0, min(5, len(kalan)))
            for __ in range(sec_say):
                if kalan:
                    t = random.choice(kalan)
                    if r.teslimat_ekle(t):
                        kalan.remove(t)

        random.shuffle(kalan)
        for t in kalan:
            random.shuffle(rotalar)
            for r in rotalar:
                if r.teslimat_ekle(t):
                    break

        populasyon.append(rotalar)

    en_iyi_fitness = float('-inf')
    en_iyi_cozum = None

    for g in range(nesil_sayisi):
        puanli = []
        for rotalar in populasyon:
            f = uygunluk_degeri(rotalar, yasak_bolgeler, simdi)
            puanli.append((f, rotalar))
            if f > en_iyi_fitness:
                en_iyi_fitness = f
                en_iyi_cozum = rotalar

        puanli.sort(key=lambda x: x[0], reverse=True)
        populasyon = [x[1] for x in puanli[:pop_boyutu // 2]]

        yeni_pop = []
        while len(yeni_pop) < pop_boyutu:
            ebeveynler = random.sample(populasyon, min(2, len(populasyon)))
            cocuk = caprazlama(ebeveynler[0], ebeveynler[1], dronlar)
            mutasyon(cocuk, teslimatlar, oran=0.2)
            yeni_pop.append(cocuk)

        populasyon = yeni_pop

        if g % 10 == 0 or g == nesil_sayisi - 1:
            print(f"Nesil {g + 1}: En İyi Uygunluk = {en_iyi_fitness:.2f}")

    for r in en_iyi_cozum:
        r.yol_hesapla(yasak_bolgeler, simdi)

    bitis = time.time()
    sure = bitis - baslangic
    print(f"Algoritma çalışma süresi: {sure:.2f} saniye")

    return en_iyi_cozum, sure


# -------------------------
# Veri Üretimi
# -------------------------

def rastgele_zaman_araligi():
    baslangic = random.randint(9, 15)
    bas = datetime.time(baslangic, 0)
    son = datetime.time(baslangic + 1, 0)
    return (bas, son)


def dron_uret(adet=5) -> List[Ucak]:
    liste = []
    for i in range(adet):
        liste.append(Ucak(
            idx=i,
            maks_agirlik=random.uniform(5, 15),
            batarya=random.randint(8000, 12000),
            hiz=random.uniform(5, 15),
            baslangic=(random.uniform(0, 100), random.uniform(0, 100))
        ))
    return liste


def teslimat_uret(adet=20) -> List[TeslimatNoktasi]:
    liste = []
    for i in range(adet):
        liste.append(TeslimatNoktasi(
            idx=i,
            konum=(random.uniform(0, 100), random.uniform(0, 100)),
            agirlik=random.uniform(0.1, 5),
            oncelik=random.randint(1, 5),
            zaman_araligi=rastgele_zaman_araligi()
        ))
    return liste


def yasak_bolge_uret(adet=2) -> List[UcusYasakBolgesi]:
    liste = []
    for i in range(adet):
        x, y = random.uniform(10, 90), random.uniform(10, 90)
        boyut = random.uniform(5, 15)
        koord = [(x, y), (x + boyut, y), (x + boyut, y + boyut), (x, y + boyut)]
        zaman = (datetime.time(0, 0), datetime.time(23, 59))
        liste.append(UcusYasakBolgesi(i, koord, zaman))
    return liste


# -------------------------
# Görselleştirme
# -------------------------

def rota_ciz(rotalar: List[Rota], yasak_bolgeler: List[UcusYasakBolgesi], teslimatlar: List[TeslimatNoktasi],
             baslik="Drone Teslimat Rotaları"):
    plt.figure(figsize=(12, 10))

    for bolge in yasak_bolgeler:
        xs, ys = zip(*bolge.koordinatlar)
        xs = list(xs) + [xs[0]]
        ys = list(ys) + [ys[0]]
        plt.fill(xs, ys, 'r', alpha=0.3)
        plt.plot(xs, ys, 'r-')

    teslim_edilen = set()
    for r in rotalar:
        for t in r.teslimatlar:
            teslim_edilen.add(t.idx)

    for t in teslimatlar:
        if t.idx not in teslim_edilen:
            plt.plot(t.konum[0], t.konum[1], 'yo', markersize=8)

    renkler = ['b', 'g', 'm', 'c']
    for i, r in enumerate(rotalar):
        if not r.gecerli or not r.yol:
            continue

        renk = renkler[i % len(renkler)]

        plt.plot(r.ucak.baslangic[0], r.ucak.baslangic[1], 'o', color=renk, markersize=10)

        for t in r.teslimatlar:
            plt.plot(t.konum[0], t.konum[1], 'o', color=renk, markersize=8)

        xs, ys = zip(*r.yol)
        plt.plot(xs, ys, '-', color=renk, label=f'Uçak {r.ucak.idx} Rotası')

    plt.plot([], [], 'r-', label='Uçuş Yasağı Bölgesi')
    plt.plot([], [], 'yo', label='Teslim Edilmemiş')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(baslik)
    plt.xlabel("X (metre)")
    plt.ylabel("Y (metre)")
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


# -------------------------
# Performans Analizi
# -------------------------

def performans_analizi(rotalar: List[Rota], teslimatlar: List[TeslimatNoktasi], calisma_suresi: float):
    teslim_edilen = set()
    for r in rotalar:
        for t in r.teslimatlar:
            teslim_edilen.add(t.idx)

    toplam_teslimat = len(teslim_edilen)
    yuzde = (toplam_teslimat / len(teslimatlar)) * 100

    gecerli_rotalar = [r for r in rotalar if r.gecerli]
    toplam_enerji = sum(r.enerji_tuketimi for r in gecerli_rotalar)
    ort_enerji = toplam_enerji / len(gecerli_rotalar) if gecerli_rotalar else 0

    toplam_mesafe = sum(r.toplam_mesafe for r in gecerli_rotalar)
    ort_mesafe = toplam_mesafe / len(gecerli_rotalar) if gecerli_rotalar else 0

    print("\nPerformans Analizi:")
    print(f"Tamamlanan teslimat sayısı: {toplam_teslimat}/{len(teslimatlar)}")
    print(f"Teslimat yüzdesi: %{yuzde:.2f}")
    print(f"Ortalama enerji tüketimi: {ort_enerji:.2f} birim")
    print(f"Ortalama mesafe: {ort_mesafe:.2f} metre")
    print(f"Algoritma çalışma süresi: {calisma_suresi:.2f} saniye")

    print("\nUçak Bazında Sonuçlar:")
    for r in rotalar:
        if r.gecerli:
            print(f"Uçak {r.ucak.idx}: {len(r.teslimatlar)} teslimat, "
                  f"{r.toplam_mesafe:.2f} metre, {r.enerji_tuketimi:.2f} enerji")

    return {
        "yuzde": yuzde,
        "ort_enerji": ort_enerji,
        "ort_mesafe": ort_mesafe,
        "calisma_suresi": calisma_suresi,
        "toplam_teslimat": toplam_teslimat
    }


# -------------------------
# Zaman Karmaşıklığı
# -------------------------

def zaman_karmaşıkligi():
    print("\nZaman Karmaşıklığı Analizi:")
    print("1. A* Algoritması: O(E log V), E = kenar sayısı, V = düğüm sayısı")
    print("2. Genetik Algoritma: O(G * P * D * A), G = nesil sayısı, P = popülasyon boyutu, D = drone sayısı, A = A* çağrı sayısı")
    print("3. Kısıt Kontrolleri: O(D * T), D = drone sayısı, T = teslimat sayısı")
    print("4. Toplam: O(G * P * D * T * E log V)")


# -------------------------
# Ana Fonksiyon
# -------------------------

def main():
    random.seed(42)

    print("Senaryo 1 Başlıyor...")
    dronlar1 = dron_uret(5)
    teslimatlar1 = teslimat_uret(20)
    yasaklar1 = yasak_bolge_uret(2)

    print(f"Dron sayısı: {len(dronlar1)}")
    print(f"Teslimat sayısı: {len(teslimatlar1)}")
    print(f"Yasak bölge sayısı: {len(yasaklar1)}")

    print("Rota optimizasyonu çalışıyor...")
    en_iyi_rotalar1, sure1 = genetik_optim(dronlar1, teslimatlar1, yasaklar1, nesil_sayisi=30, pop_boyutu=20)

    performans_analizi(en_iyi_rotalar1, teslimatlar1, sure1)
    print("Rotalar görselleştiriliyor...")
    rota_ciz(en_iyi_rotalar1, yasaklar1, teslimatlar1, "Senaryo 1: 5 Drone, 20 Teslimat, 2 Yasak Bölge")

    print("\nSenaryo 2 Başlıyor...")
    dronlar2 = dron_uret(10)
    teslimatlar2 = teslimat_uret(50)
    yasaklar2 = yasak_bolge_uret(5)

    print(f"Dron sayısı: {len(dronlar2)}")
    print(f"Teslimat sayısı: {len(teslimatlar2)}")
    print(f"Yasak bölge sayısı: {len(yasaklar2)}")

    print("Rota optimizasyonu çalışıyor...")
    en_iyi_rotalar2, sure2 = genetik_optim(dronlar2, teslimatlar2, yasaklar2, nesil_sayisi=30, pop_boyutu=20)

    performans_analizi(en_iyi_rotalar2, teslimatlar2, sure2)
    print("Rotalar görselleştiriliyor...")
    rota_ciz(en_iyi_rotalar2, yasaklar2, teslimatlar2, "Senaryo 2: 10 Drone, 50 Teslimat, 5 Yasak Bölge")

    zaman_karmaşıkligi()


if __name__ == "__main__":
    main()
