"""
recommend.py
------------
Rule-based skincare product recommendation engine.
Maps predicted skin condition + skin type cluster → product recommendations.
"""

from src.data_loader import CLASS_NAMES

# ─────────────────────────────────────────────────────────────
# Skin condition → skincare concern mapping
# ─────────────────────────────────────────────────────────────
CONDITION_CONCERN = {
    "Psoriasis":               "Peradangan & Pengelupasan Kronis",
    "Seboreic Dermatitis":     "Kulit Berminyak & Berketombe",
    "Lichen Planus":           "Lesi & Peradangan",
    "Pityriasis Rosea":        "Ruam & Iritasi Sementara",
    "Chronic Dermatitis":      "Eksim & Kulit Sensitif Kronis",
    "Pityriasis Rubra Pilaris":"Keratinisasi Abnormal",
}

CONDITION_AVOID = {
    "Psoriasis":               ["Alkohol denat", "Parfum sintetis", "SLS/SLES", "Retinol dosis tinggi"],
    "Seboreic Dermatitis":     ["Minyak kelapa (comedogenic)", "Produk terlalu oklusif", "Emolien berat"],
    "Lichen Planus":           ["Pewarna buatan", "Alkohol", "Fragrance"],
    "Pityriasis Rosea":        ["Panas berlebih", "Produk eksfoliasi keras", "Alkohol"],
    "Chronic Dermatitis":      ["SLS/SLES", "Fragrance", "Pengawet keras (parabens)", "Alkohol denat"],
    "Pityriasis Rubra Pilaris":["Retinoid OTC", "Eksfoliator fisik", "Produk berbasis alkohol"],
}

# ─────────────────────────────────────────────────────────────
# Skin type cluster → name & description
# ─────────────────────────────────────────────────────────────
SKIN_PROFILES = {
    0: {
        "name": "Kulit Sensitif",
        "icon": "🌸",
        "desc": "Kulit reaktif, mudah kemerahan, dan mudah iritasi terhadap produk baru.",
    },
    1: {
        "name": "Kulit Normal",
        "icon": "✨",
        "desc": "Kulit seimbang, tidak terlalu berminyak atau kering, pori-pori tidak terlihat jelas.",
    },
    2: {
        "name": "Kulit Berminyak",
        "icon": "💧",
        "desc": "Produksi sebum berlebih, pori-pori terlihat besar, rentan terhadap jerawat.",
    },
    3: {
        "name": "Kulit Kering",
        "icon": "🍂",
        "desc": "Kulit terasa ketat, kasar, atau bersisik; kurang kelembapan alami.",
    },
    4: {
        "name": "Kulit Kombinasi",
        "icon": "🌗",
        "desc": "T-zone berminyak (dahi, hidung, dagu) namun pipi cenderung normal/kering.",
    },
}

# ─────────────────────────────────────────────────────────────
# Product database per (condition, skin_type)
# ─────────────────────────────────────────────────────────────
PRODUCT_DB = {
    # ─── CLEANSER ───────────────────────────────────────────
    "cleanser": {
        "Psoriasis": {
            "default": {
                "name": "CeraVe Hydrating Cleanser",
                "brand": "CeraVe",
                "ingredients": ["Ceramide NP", "Ceramide AP", "Hyaluronic Acid", "Niacinamide"],
                "why": "Formula bebas sabun & fragrance, mempertahankan skin barrier yang sering terganggu pada psoriasis.",
            }
        },
        "Seboreic Dermatitis": {
            "default": {
                "name": "Neutrogena T/Gel Shampoo (sebagai face wash)",
                "brand": "Neutrogena",
                "ingredients": ["Coal Tar 1%", "Zinc Pyrithione"],
                "why": "Mengontrol pertumbuhan Malassezia yang memicu seboreic dermatitis.",
            }
        },
        "Lichen Planus": {
            "default": {
                "name": "La Roche-Posay Toleriane Hydrating Gentle Cleanser",
                "brand": "La Roche-Posay",
                "ingredients": ["Niacinamide", "Glycerin", "Ceramide"],
                "why": "Sangat lembut, bebas fragrance, cocok untuk kulit dengan lesi dan peradangan.",
            }
        },
        "Pityriasis Rosea": {
            "default": {
                "name": "Dove Sensitive Skin Beauty Bar",
                "brand": "Dove",
                "ingredients": ["Sodium Lauroyl Isethionate", "Stearic Acid"],
                "why": "pH-balanced, lembut, tidak memperparah iritasi sementara.",
            }
        },
        "Chronic Dermatitis": {
            "default": {
                "name": "Avène XeraCalm A.D Lipid-Replenishing Cleansing Oil",
                "brand": "Avène",
                "ingredients": ["Diacylglycerol", "Avène Thermal Spring Water"],
                "why": "Membersihkan tanpa mengganggu lipid barrier, ideal untuk eksim kronis.",
            }
        },
        "Pityriasis Rubra Pilaris": {
            "default": {
                "name": "Vanicream Gentle Facial Cleanser",
                "brand": "Vanicream",
                "ingredients": ["Glycerin", "Hydroxyethylcellulose"],
                "why": "Bebas pewarna, fragrance, dan bahan iritan — aman untuk kondisi kulit langka ini.",
            }
        },
    },

    # ─── TONER / ESSENCE ────────────────────────────────────
    "toner": {
        "Psoriasis": {
            "default": {
                "name": "COSRX AHA/BHA Clarifying Treatment Toner",
                "brand": "COSRX",
                "ingredients": ["Betaine Salicylate", "Niacinamide"],
                "why": "Eksfoliasi lembut untuk kulit bersisik, tidak iritan.",
            }
        },
        "Seboreic Dermatitis": {
            "default": {
                "name": "Paula's Choice Skin Balancing Pore-Reducing Toner",
                "brand": "Paula's Choice",
                "ingredients": ["Niacinamide", "Panthenol", "Glycerin"],
                "why": "Mengontrol minyak berlebih, memperkecil pori, dan menenangkan kulit.",
            }
        },
        "Lichen Planus": {
            "default": {
                "name": "Klairs Supple Preparation Unscented Toner",
                "brand": "Klairs",
                "ingredients": ["Centella Asiatica", "Hyaluronic Acid", "Beta-Glucan"],
                "why": "Bebas parfum, menenangkan, dan melembapkan tanpa memperparah lesi.",
            }
        },
        "Pityriasis Rosea": {
            "default": {
                "name": "Hada Labo Gokujyun Premium Lotion",
                "brand": "Hada Labo",
                "ingredients": ["5 Jenis Hyaluronic Acid", "Collagen"],
                "why": "Hidrasi mendalam, membantu pemulihan skin barrier selama fase erupsi.",
            }
        },
        "Chronic Dermatitis": {
            "default": {
                "name": "Avène Thermal Spring Water Spray",
                "brand": "Avène",
                "ingredients": ["Avène Thermal Spring Water"],
                "why": "Menenangkan seketika, bebas bahan aktif yang bisa mengiritasi.",
            }
        },
        "Pityriasis Rubra Pilaris": {
            "default": {
                "name": "Physiogel AI Toner",
                "brand": "Physiogel",
                "ingredients": ["BioMimic Technology", "Ceramide"],
                "why": "Merestorasi lipid barrier, sangat lembut untuk kulit dengan kondisi langka.",
            }
        },
    },

    # ─── MOISTURIZER ────────────────────────────────────────
    "moisturizer": {
        "Psoriasis": {
            "default": {
                "name": "CeraVe Moisturizing Cream",
                "brand": "CeraVe",
                "ingredients": ["Ceramide NP/AP/EOP", "Hyaluronic Acid", "Dimethicone"],
                "why": "Memulihkan skin barrier yang rusak, cocok untuk kulit bersisik dan meradang.",
            }
        },
        "Seboreic Dermatitis": {
            "default": {
                "name": "Neutrogena Hydro Boost Water Gel",
                "brand": "Neutrogena",
                "ingredients": ["Hyaluronic Acid", "Dimethicone"],
                "why": "Pelembap berbasis air, ringan, tidak menyumbat pori — ideal untuk kulit berminyak.",
            }
        },
        "Lichen Planus": {
            "default": {
                "name": "Eucerin Original Healing Cream",
                "brand": "Eucerin",
                "ingredients": ["Mineral Oil", "Lanolin Alcohol", "Panthenol"],
                "why": "Melembapkan intensif area dengan lesi, bebas pewangi.",
            }
        },
        "Pityriasis Rosea": {
            "default": {
                "name": "Aveeno Daily Moisturizing Lotion",
                "brand": "Aveeno",
                "ingredients": ["Colloidal Oatmeal 1%", "Dimethicone", "Glycerin"],
                "why": "Oatmeal koloid mengurangi gatal dan menenangkan kulit yang ruam.",
            }
        },
        "Chronic Dermatitis": {
            "default": {
                "name": "La Roche-Posay Lipikar Balm AP+M",
                "brand": "La Roche-Posay",
                "ingredients": ["Niacinamide", "Shea Butter", "Ceramide"],
                "why": "Diformulasikan khusus untuk eksim, memulihkan microbiome kulit.",
            }
        },
        "Pityriasis Rubra Pilaris": {
            "default": {
                "name": "Aquaphor Healing Ointment",
                "brand": "Aquaphor",
                "ingredients": ["Petrolatum 41%", "Panthenol", "Glycerin", "Bisabolol"],
                "why": "Oklusif kuat, melindungi dan melembapkan ekstrem untuk kulit hiperatosis.",
            }
        },
    },

    # ─── SUNSCREEN ──────────────────────────────────────────
    "sunscreen": {
        "Psoriasis": {
            "default": {
                "name": "EltaMD UV Clear Broad-Spectrum SPF 46",
                "brand": "EltaMD",
                "ingredients": ["Zinc Oxide 9%", "Octinoxate", "Niacinamide"],
                "why": "Zinc fisik membantu mengurangi peradangan, bebas fragrance.",
            }
        },
        "Seboreic Dermatitis": {
            "default": {
                "name": "Biore UV Aqua Rich Watery Essence SPF 50+",
                "brand": "Biore",
                "ingredients": ["UV Filter", "Micro Defense", "Hyaluronic Acid"],
                "why": "Tekstur sangat ringan seperti air, tidak memperparah kulit berminyak.",
            }
        },
        "Lichen Planus": {
            "default": {
                "name": "Avène Very High Protection Mineral Fluid SPF 50+",
                "brand": "Avène",
                "ingredients": ["Titanium Dioxide", "Zinc Oxide"],
                "why": "100% mineral filter, lembut untuk kulit dengan lesi dan sangat sensitif.",
            }
        },
        "Pityriasis Rosea": {
            "default": {
                "name": "La Roche-Posay Anthelios Mineral SPF 50",
                "brand": "La Roche-Posay",
                "ingredients": ["Titanium Dioxide", "Mexoryl XL"],
                "why": "Mineral sunscreen yang menenangkan, mencegah sinar UV memperparah ruam.",
            }
        },
        "Chronic Dermatitis": {
            "default": {
                "name": "Vanicream Mineral Sunscreen SPF 50",
                "brand": "Vanicream",
                "ingredients": ["Zinc Oxide 16%"],
                "why": "Bebas semua iritan umum, aman untuk eksim parah.",
            }
        },
        "Pityriasis Rubra Pilaris": {
            "default": {
                "name": "Blue Lizard Sensitive Mineral Sunscreen SPF 30+",
                "brand": "Blue Lizard",
                "ingredients": ["Zinc Oxide", "Titanium Dioxide"],
                "why": "Hypoallergenic, bebas chemical filter yang bisa mengiritasi.",
            }
        },
    },

    # ─── SERUM / TREATMENT ──────────────────────────────────
    "serum": {
        "Psoriasis": {
            "default": {
                "name": "The Ordinary Niacinamide 10% + Zinc 1%",
                "brand": "The Ordinary",
                "ingredients": ["Niacinamide 10%", "Zinc PCA 1%"],
                "why": "Mengurangi kemerahan, mengontrol produksi minyak, memperkuat barrier.",
            }
        },
        "Seboreic Dermatitis": {
            "default": {
                "name": "Inkey List Salicylic Acid Cleanser + 2% BHA Serum",
                "brand": "The Inkey List",
                "ingredients": ["Salicylic Acid 2%", "Zinc PCA"],
                "why": "BHA menembus pori, mengontrol Malassezia dan mengurangi sisik.",
            }
        },
        "Lichen Planus": {
            "default": {
                "name": "Dr. Jart+ Cicapair Tiger Grass Serum",
                "brand": "Dr. Jart+",
                "ingredients": ["Centella Asiatica", "Madecassoside", "Chlorophyll"],
                "why": "Centella menenangkan peradangan dan mempercepat pemulihan lesi.",
            }
        },
        "Pityriasis Rosea": {
            "default": {
                "name": "COSRX Advanced Snail 96 Mucin Power Essence",
                "brand": "COSRX",
                "ingredients": ["Snail Secretion Filtrate 96%", "Sodium Hyaluronate"],
                "why": "Mempercepat penyembuhan, regenerasi kulit, dan menenangkan iritasi.",
            }
        },
        "Chronic Dermatitis": {
            "default": {
                "name": "Avène Cicalfate+ Restorative Protective Cream",
                "brand": "Avène",
                "ingredients": ["Sucralfate", "Copper-Zinc", "Avène Thermal Water"],
                "why": "Dikhususkan untuk memperbaiki skin barrier pada eksim dan dermatitis.",
            }
        },
        "Pityriasis Rubra Pilaris": {
            "default": {
                "name": "SkinCeuticals Silymarin CF (Vitamin C)",
                "brand": "SkinCeuticals",
                "ingredients": ["Silymarin", "Vitamin C 15%", "Ferulic Acid"],
                "why": "Antioksidan kuat, membantu mengurangi hiperkeratosis dan kerusakan oksidatif.",
            }
        },
    },
}

# ─────────────────────────────────────────────────────────────
# TIPS UMUM per kondisi
# ─────────────────────────────────────────────────────────────
CONDITION_TIPS = {
    "Psoriasis": [
        "Mandi dengan air hangat (bukan panas) untuk mencegah kekeringan.",
        "Oleskan pelembap segera setelah mandi selagi kulit masih lembap.",
        "Gunakan humidifier di kamar untuk menjaga kelembapan udara.",
        "Hindari stres berlebih — stres adalah pemicu utama flare psoriasis.",
        "Konsultasikan dengan dokter kulit untuk terapi sistemik jika perlu.",
    ],
    "Seboreic Dermatitis": [
        "Cuci wajah 2x sehari dengan pembersih antijamur.",
        "Hindari produk berbasis minyak berat yang bisa memperparah Malassezia.",
        "Gunakan shampoo antijamur 1-2x seminggu jika area kepala terdampak.",
        "Kelola stres dan pola tidur yang baik untuk mencegah flare.",
        "Jaga kulit kepala tetap bersih dan kering.",
    ],
    "Lichen Planus": [
        "Hindari produk dengan fragrance dan alkohol.",
        "Aplikasikan emolien lembut secara teratur.",
        "Lindungi kulit dari trauma fisik (scratching) yang bisa memicu lesi baru.",
        "Konsultasikan dokter untuk terapi kortikosteroid topikal jika diperlukan.",
    ],
    "Pityriasis Rosea": [
        "Kondisi ini biasanya sembuh sendiri dalam 6-12 minggu.",
        "Hindari mandi air panas yang bisa memperparah gatal.",
        "Gunakan pakaian berbahan katun yang longgar.",
        "Lindungi kulit dari paparan sinar matahari langsung.",
        "Gunakan lotion calamine untuk mengurangi gatal.",
    ],
    "Chronic Dermatitis": [
        "Identifikasi dan hindari alergen/iritan pemicu eksim Anda.",
        "Mandi singkat dengan air hangat, tidak lebih dari 10 menit.",
        "Pelembap harus diaplikasikan minimal 2x sehari.",
        "Potong kuku pendek untuk mencegah kerusakan akibat garukan.",
        "Gunakan detergen bebas pewangi untuk pakaian.",
    ],
    "Pityriasis Rubra Pilaris": [
        "Kondisi langka ini memerlukan penanganan spesialis dermatologi.",
        "Jaga kelembapan kulit dengan emolien sangat oklusif.",
        "Hindari paparan sinar matahari berlebih.",
        "Pakaikan pakaian lembut berbahan katun untuk mengurangi gesekan.",
        "Pertimbangkan terapi retinoid sistemik yang diresepkan dokter.",
    ],
}


def get_recommendation(condition_name: str, cluster_id: int, age: int = 25):
    """
    Build a full recommendation dictionary based on predicted condition and cluster.

    Args:
        condition_name: Predicted condition name string (from CLASS_NAMES mapping)
        cluster_id: K-Means cluster integer (0-4)
        age: User's age

    Returns:
        dict with keys: condition, concern, skin_profile, products, avoid, tips
    """
    skin_profile = SKIN_PROFILES.get(cluster_id, SKIN_PROFILES[1])
    concern      = CONDITION_CONCERN.get(condition_name, "Masalah Kulit Umum")
    avoid        = CONDITION_AVOID.get(condition_name, [])
    tips         = CONDITION_TIPS.get(condition_name, [])

    products = {}
    cat_db   = PRODUCT_DB
    for category in ["cleanser", "toner", "moisturizer", "sunscreen", "serum"]:
        cond_map = cat_db[category].get(condition_name, {})
        prod     = cond_map.get("default", None)
        if prod:
            products[category] = prod

    return {
        "condition":    condition_name,
        "concern":      concern,
        "skin_profile": skin_profile,
        "products":     products,
        "avoid":        avoid,
        "tips":         tips,
        "age_note":     "Untuk usia muda (18-35 tahun), fokus pada pencegahan dini & perlindungan."
                        if 18 <= age <= 35 else "",
    }
