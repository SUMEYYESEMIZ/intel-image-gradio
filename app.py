import gradio as gr
import numpy as np
from PIL import Image

CLASSES = ["cadde", "deniz", "orman", "dag", "ic_mekan", "tarihi_yer"]

def predict(img: Image.Image):
    # Çok hızlı "baseline" (renk + kontrast heuristik)
    im = img.convert("RGB").resize((256, 256))
    arr = np.asarray(im).astype(np.float32) / 255.0

    r, g, b = arr[...,0].mean(), arr[...,1].mean(), arr[...,2].mean()
    bright = arr.mean()
    contrast = arr.std()

    # Basit skorlar (tam bilimsel değil ama demo için çalışır)
    scores = {c: 0.0 for c in CLASSES}

    # Deniz: mavi baskın
    scores["deniz"] += (b - (r+g)/2) * 3.0

    # Orman: yeşil baskın
    scores["orman"] += (g - (r+b)/2) * 3.0

    # İç mekan: daha düşük mavi/yeşil, orta parlaklık + düşük kontrast
    scores["ic_mekan"] += (0.6 - abs(bright-0.55)) + (0.15 - contrast)

    # Dağ: yüksek kontrast + düşük parlaklık eğilimi
    scores["dag"] += (contrast*2.0) + (0.55 - bright)

    # Cadde: orta kontrast + kırmızı/yeşil dengeli (nötr)
    scores["cadde"] += (0.4 - abs((r+g+b)/3 - 0.45)) + (0.25 - abs(contrast-0.18))

    # Tarihi yer: genelde sıcak tonlar (kırmızı + sarı)
    scores["tarihi_yer"] += (r + g - b) * 1.5

    # Normalize
    vals = np.array(list(scores.values()), dtype=np.float32)
    # softmax
    exps = np.exp(vals - vals.max())
    probs = exps / exps.sum()

    prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    top = max(prob_dict, key=prob_dict.get)
    return top, prob_dict

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Görsel Yükle"),
    outputs=[
        gr.Label(label="Tahmin (6 sınıf)"),
        gr.JSON(label="Olasılıklar")
    ],
    title="6 Sınıflı Sahne Tanıma (Demo)",
    description="Cadde / Deniz / Orman / Dağ / İç Mekan / Tarihi Yer için hızlı baseline demo (heuristic)."
)

if __name__ == "__main__":
    demo.launch()
