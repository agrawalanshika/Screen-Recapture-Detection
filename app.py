import os
import io
import base64
import numpy as np
import cv2
import joblib
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

from feature_engineering.features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_train", "best_random_forest.pkl")
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    "Blur Score", "Edge Density", "Brightness", "Entropy",
    "Mean Red", "Mean Green", "Mean Blue",
    "Std Red", "Std Green", "Std Blue",
    "Sobel X", "Sobel Y",
    "FFT High", "FFT Low",
    "LBP Mean", "LBP Variance",
    "RMS Contrast", "Saturation Mean",
    "Tenengrad", "Noise Estimate", "Pixel Grid Score"
]

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Screen Recapture Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <!-- Aptos itself is a Microsoft-licensed font and isn't available on a public CDN,
       so Hanken Grotesk is used here — the closest free match: the same humanist,
       rounded-but-geometric proportions Aptos is known for. -->
  <link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:            #08080b;
      --bg-grid:       #111118;
      --panel:         #121217;
      --panel-2:       #15151c;
      --border:        #232330;
      --border-soft:   #1a1a22;
      --text:          #f1f1f4;
      --muted:         #87878f;
      --muted-2:       #56565f;
      --accent:        #7c6cff;
      --accent-dim:    #4d3fc7;
      --real:          #34e7a6;
      --real-bg:       #0a2420;
      --real-border:   #1f5945;
      --screen:        #ff5a72;
      --screen-bg:     #2a0e17;
      --screen-border: #5c1f30;
      --font-display:  'Hanken Grotesk', sans-serif;
      --font-body:     'Hanken Grotesk', sans-serif;
      --font-mono:     'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      height: 100%;
    }

    body {
      font-family: var(--font-body);
      background:
        radial-gradient(circle at 50% -10%, rgba(124,108,255,0.10), transparent 55%),
        var(--bg);
      background-attachment: fixed;
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 48px 18px 32px;
      -webkit-font-smoothing: antialiased;
    }

    .eyebrow {
      font-family: var(--font-mono);
      font-size: 0.68rem;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 14px;
    }

    .eyebrow .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px 1px var(--accent);
    }

    h1 {
      font-family: var(--font-display);
      font-size: 2.6rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 10px;
      text-align: center;
    }

    .subtitle {
      font-size: 0.86rem;
      color: var(--muted);
      margin-bottom: 56px;
      text-align: center;
      max-width: 360px;
      line-height: 1.5;
    }

    .card {
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 22px;
      width: 100%;
      max-width: 880px;
      box-shadow: 0 30px 60px -25px rgba(0,0,0,0.6);
    }

    /* two-column on wider viewports: media on the left, controls + result on the right */
    .card-grid {
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    @media (min-width: 720px) {
      .card-grid {
        flex-direction: row;
        align-items: stretch;
        gap: 24px;
      }
      .col-left, .col-right {
        flex: 1 1 0;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      .col-right { justify-content: center; }
    }

    .col-left, .col-right {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .media-wrapper {
      position: relative;
      width: 100%;
      border-radius: 14px;
      overflow: hidden;
      background: #000;
      aspect-ratio: 4/3;
      border: 1px solid var(--border-soft);
    }

    #video, #preview {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    #preview { display: none; }

    .bracket {
      position: absolute;
      width: 22px;
      height: 22px;
      border: 2px solid var(--accent);
      opacity: 0.85;
      pointer-events: none;
      z-index: 3;
      filter: drop-shadow(0 0 4px rgba(124,108,255,0.5));
    }
    .bracket.tl { top: 10px;    left: 10px;    border-right: none;  border-bottom: none; border-top-left-radius: 6px; }
    .bracket.tr { top: 10px;    right: 10px;   border-left: none;   border-bottom: none; border-top-right-radius: 6px; }
    .bracket.bl { bottom: 10px; left: 10px;    border-right: none;  border-top: none;    border-bottom-left-radius: 6px; }
    .bracket.br { bottom: 10px; right: 10px;   border-left: none;   border-top: none;    border-bottom-right-radius: 6px; }

    .scan-line {
      position: absolute;
      left: 0; right: 0;
      height: 2px;
      top: 0;
      background: linear-gradient(90deg, transparent, var(--accent) 20%, #d6d1ff 50%, var(--accent) 80%, transparent);
      box-shadow: 0 0 12px 2px rgba(124,108,255,0.7);
      opacity: 0;
      z-index: 4;
      pointer-events: none;
    }
    .media-wrapper.scanning .scan-line {
      animation: sweep 1.4s ease-in-out infinite;
    }
    @keyframes sweep {
      0%   { top: 4%;  opacity: 0; }
      8%   { opacity: 1; }
      50%  { top: 96%; opacity: 1; }
      92%  { opacity: 1; }
      100% { top: 4%;  opacity: 0; }
    }

    #switchBtn {
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 5;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: rgba(10,10,14,0.55);
      border: 1px solid rgba(255,255,255,0.15);
      backdrop-filter: blur(6px);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: background 0.15s, transform 0.15s;
      flex: none;
      padding: 0;
    }
    #switchBtn:hover { background: rgba(124,108,255,0.35); }
    #switchBtn:active { transform: scale(0.92); }
    #switchBtn svg { width: 18px; height: 18px; stroke: #fff; }
    #switchBtn.spin svg { animation: spin360 0.5s ease; }
    @keyframes spin360 { to { transform: rotate(180deg); } }

    .media-wrapper.has-preview #switchBtn { display: none; }
    .media-wrapper.has-preview .bracket { opacity: 0.4; }


    .btn-row {
      display: flex;
      gap: 10px;
      width: 100%;
    }

    button {
      flex: 1;
      padding: 13px;
      border: none;
      border-radius: 11px;
      font-family: var(--font-display);
      font-size: 0.92rem;
      font-weight: 600;
      letter-spacing: -0.01em;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s, box-shadow 0.15s;
    }

    button:active { transform: scale(0.985); }
    button:disabled { opacity: 0.35; cursor: not-allowed; }

    #captureBtn {
      background: var(--accent);
      color: #fff;
      box-shadow: 0 8px 20px -8px rgba(124,108,255,0.7);
    }
    #captureBtn:hover { background: #8c7eff; }

    #retakeBtn {
      background: var(--panel-2);
      border: 1px solid var(--border);
      color: var(--muted);
      display: none;
    }
    #retakeBtn:hover { color: var(--text); }

    #analyseBtn {
      background: var(--real);
      color: #06140f;
      display: none;
      box-shadow: 0 8px 20px -8px rgba(52,231,166,0.5);
    }
    #analyseBtn:hover { background: #54f0bb; }

    .upload-strip {
      width: 100%;
      border: 1.5px dashed var(--border);
      border-radius: 12px;
      padding: 15px;
      text-align: center;
      color: var(--muted-2);
      font-size: 0.82rem;
      font-family: var(--font-display);
      letter-spacing: 0.01em;
      cursor: pointer;
      transition: border-color 0.15s, color 0.15s, background 0.15s;
    }
    .upload-strip:hover {
      border-color: var(--accent);
      color: var(--muted);
      background: rgba(124,108,255,0.05);
    }
    #fileInput { display: none; }

    #result {
      width: 100%;
      border-radius: 14px;
      padding: 20px 22px;
      display: none;
      flex-direction: column;
      gap: 12px;
      animation: rise 0.3s ease;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .result-real   { background: var(--real-bg);   border: 1px solid var(--real-border); }
    .result-screen { background: var(--screen-bg); border: 1px solid var(--screen-border); }

    .result-label {
      font-family: var(--font-display);
      font-size: 1.3rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }

    .real-color   { color: var(--real); }
    .screen-color { color: var(--screen); }

    .score-row {
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: var(--font-display);
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--muted);
    }

    .bar-bg {
      flex: 1;
      height: 5px;
      background: rgba(255,255,255,0.08);
      border-radius: 99px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 99px;
      transition: width 0.5s cubic-bezier(.4,0,.2,1);
    }

    .meta {
      font-family: var(--font-display);
      font-size: 0.78rem;
      font-weight: 500;
      color: var(--muted-2);
      letter-spacing: 0.01em;
    }

    .spinner {
      width: 20px; height: 20px;
      border: 2.5px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      display: none;
      margin: 0 auto;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    footer {
      margin-top: 28px;
      font-family: var(--font-mono);
      font-size: 0.7rem;
      color: var(--muted-2);
      letter-spacing: 0.04em;
      text-align: center;
    }
  </style>
</head>
<body>

  
  <h1>Screen Recapture Detector</h1>
  <p class="subtitle">Capture a photo and it checks whether it's a genuine shot or a recapture of a screen.</p>

  <div class="card">
    <div class="card-grid">

      <!-- Left column: camera / preview + capture controls -->
      <div class="col-left">
        <div class="media-wrapper" id="mediaWrapper">
          <video id="video" autoplay playsinline muted></video>
          <img id="preview" alt="captured frame" />

          <div class="bracket tl"></div>
          <div class="bracket tr"></div>
          <div class="bracket bl"></div>
          <div class="bracket br"></div>
          <div class="scan-line"></div>

          <button id="switchBtn" title="Switch camera" aria-label="Switch camera">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M17 2l4 4-4 4"></path>
              <path d="M21 6H7a4 4 0 0 0-4 4v1"></path>
              <path d="M7 22l-4-4 4-4"></path>
              <path d="M3 18h14a4 4 0 0 0 4-4v-1"></path>
            </svg>
          </button>
        </div>

        <div class="btn-row">
          <button id="captureBtn">📸 Capture</button>
          <button id="retakeBtn">↩ Retake</button>
          <button id="analyseBtn">🔍 Analyse</button>
        </div>
      </div>

      <!-- Right column: upload fallback + spinner + result -->
      <div class="col-right">
        <div class="upload-strip" onclick="document.getElementById('fileInput').click()">
          or click to upload an image
        </div>
        <input type="file" id="fileInput" accept="image/*" />

        <div class="spinner" id="spinner"></div>

        <div id="result">
          <div class="result-label" id="labelText"></div>
          <div class="score-row">
            <span id="scoreText"></span>
            <div class="bar-bg">
              <div class="bar-fill" id="barFill"></div>
            </div>
          </div>
          <div class="meta" id="metaText"></div>
        </div>
      </div>

    </div>
  </div>


  <canvas id="canvas" style="display:none"></canvas>

  <script>
    const video        = document.getElementById('video');
    const preview      = document.getElementById('preview');
    const canvas       = document.getElementById('canvas');
    const mediaWrapper = document.getElementById('mediaWrapper');
    const captureBtn   = document.getElementById('captureBtn');
    const retakeBtn    = document.getElementById('retakeBtn');
    const analyseBtn   = document.getElementById('analyseBtn');
    const switchBtn    = document.getElementById('switchBtn');
    const fileInput    = document.getElementById('fileInput');
    const spinner      = document.getElementById('spinner');
    const result       = document.getElementById('result');
    const labelText    = document.getElementById('labelText');
    const scoreText    = document.getElementById('scoreText');
    const barFill      = document.getElementById('barFill');
    const metaText     = document.getElementById('metaText');

    let capturedDataUrl  = null;
    let currentStream    = null;
    let currentFacing    = 'environment';

    function startCamera(facingMode) {
      if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
      }
      navigator.mediaDevices.getUserMedia({ video: { facingMode }, audio: false })
        .then(stream => {
          currentStream = stream;
          video.srcObject = stream;
          captureBtn.disabled = false;
          captureBtn.textContent = '📸 Capture';
        })
        .catch(() => {
          // Fall back to default camera if the requested facing mode fails
          if (facingMode !== undefined) {
            navigator.mediaDevices.getUserMedia({ video: true, audio: false })
              .then(stream => { currentStream = stream; video.srcObject = stream; })
              .catch(() => {
                captureBtn.disabled = true;
                captureBtn.textContent = 'Camera unavailable — use upload';
              });
          }
        });
    }

    startCamera(currentFacing);

    switchBtn.addEventListener('click', () => {
      switchBtn.classList.add('spin');
      setTimeout(() => switchBtn.classList.remove('spin'), 500);
      currentFacing = currentFacing === 'environment' ? 'user' : 'environment';
      startCamera(currentFacing);
    });

    captureBtn.addEventListener('click', () => {
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      canvas.getContext('2d').drawImage(video, 0, 0);
      capturedDataUrl = canvas.toDataURL('image/jpeg', 0.9);

      preview.src     = capturedDataUrl;
      video.style.display   = 'none';
      preview.style.display = 'block';
      mediaWrapper.classList.add('has-preview');

      captureBtn.style.display  = 'none';
      retakeBtn.style.display   = 'flex';
      analyseBtn.style.display  = 'flex';
      hideResult();
    });

    retakeBtn.addEventListener('click', () => {
      capturedDataUrl = null;
      preview.style.display = 'none';
      video.style.display   = 'block';
      mediaWrapper.classList.remove('has-preview');
      captureBtn.style.display = 'flex';
      retakeBtn.style.display  = 'none';
      analyseBtn.style.display = 'none';
      hideResult();
    });


    analyseBtn.addEventListener('click', () => {
      if (capturedDataUrl) sendBase64(capturedDataUrl);
    });

    fileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = ev => {
        capturedDataUrl = ev.target.result;
        preview.src = capturedDataUrl;
        video.style.display   = 'none';
        preview.style.display = 'block';
        mediaWrapper.classList.add('has-preview');
        captureBtn.style.display  = 'none';
        retakeBtn.style.display   = 'flex';
        analyseBtn.style.display  = 'flex';
        hideResult();
        sendBase64(capturedDataUrl);   // auto-analyse on upload
      };
      reader.readAsDataURL(file);
    });

   
    function sendBase64(dataUrl) {
      spinner.style.display = 'block';
      analyseBtn.disabled   = true;
      mediaWrapper.classList.add('scanning');
      hideResult();

      fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      })
      .then(r => r.json())
      .then(data => {
        spinner.style.display = 'none';
        analyseBtn.disabled   = false;
        mediaWrapper.classList.remove('scanning');
        showResult(data);
      })
      .catch(() => {
        spinner.style.display = 'none';
        analyseBtn.disabled   = false;
        mediaWrapper.classList.remove('scanning');
        alert('Server error — make sure app.py is running.');
      });
    }

    function showResult(data) {
      const isScreen = data.label === 'SCREEN';
      const pct      = (data.score * 100).toFixed(1);

      result.className = isScreen ? 'result-screen' : 'result-real';
      labelText.className = isScreen ? 'result-label screen-color' : 'result-label real-color';
      labelText.textContent = isScreen ? '🖥️  SCREEN RECAPTURE' : '✅  REAL PHOTO';

      scoreText.textContent = `Prediction Score: ${(pct / 100).toFixed(2)}`;
      barFill.style.width      = pct + '%';
      barFill.style.background = isScreen ? 'var(--screen)' : 'var(--real)';
      metaText.textContent     = `Inference Time: ${data.latency_ms} ms `;

      result.style.display = 'flex';
    }

    function hideResult() {
      result.style.display = 'none';
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict_route():
    import time

    data_url = request.json.get("image", "")

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)
    img_np = cv2.resize(img_np, (256, 256))

    t0 = time.perf_counter()
    features = extract_features(img_np)
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    score = float(model.predict_proba(features_df)[0][1])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    label = "SCREEN" if score >= 0.5 else "REAL"

    return jsonify({
        "score": round(score, 4),
        "label": label,
        "latency_ms": latency_ms
    })


if __name__ == "__main__":
    print("\n  Screen Recapture Detector — live demo")
    print("  Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)