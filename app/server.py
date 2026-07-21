import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from flask import Flask, request, jsonify, render_template, send_file
from pathlib import Path
import zipfile
import tempfile
import shutil
import torch
import numpy as np
from PIL import Image

from model import build_model
from train import eval_transforms
from breed_records import get_formula_vector_for_breed, FOLDER_TO_BREEDS
from formulas import get_formula_vector
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

# ── Setup ─────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload

MODELS_DIR = Path(__file__).parent.parent / 'models'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Load model once at startup
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

print("Loading Green Cattle model...")
model = build_model(freeze_backbone=False)
model.load_state_dict(
    torch.load(MODELS_DIR / 'green_cattle_best.pth',
               map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ── Helper functions ──────────────────────────────────────────────
def score_image(img_path, formula_vec=None):
    """
    Score one cow image.
    If formula_vec is provided (real farm records), use it.
    Otherwise fall back to visual-only scoring with a neutral formula vector.
    """
    image = Image.open(img_path).convert('RGB')
    image_tensor = eval_transforms(image).unsqueeze(0).to(device)

    if formula_vec is None:
        # No farm records — use a neutral mid-range formula vector
        # This represents an "average unknown" cow
        neutral_vec = np.full(23, 380.0, dtype=np.float32)
        formula_tensor = torch.tensor(neutral_vec).unsqueeze(0).to(device)
    else:
        formula_tensor = torch.tensor(
            np.array(formula_vec, dtype=np.float32)
        ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor, formula_tensor)
        probs = torch.softmax(output, dim=1)

    low_ch4_prob = probs[0][1].item()
    confidence = probs.max().item()
    return low_ch4_prob, confidence


def predict_offspring(score1, score2, n_simulations=100):
    """
    Simulate offspring CH4 score using additive genetic model.
    Runs multiple simulations and returns mean + std.
    """
    scores = []
    for _ in range(n_simulations):
        base = (score1 + score2) / 2
        noise = np.random.normal(0, 0.04)
        scores.append(float(np.clip(base + noise, 0, 1)))
    return np.mean(scores), np.std(scores)


# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Step 1: receive ZIP file, extract images, score each one visually.
    Returns list of cows with their visual-only scores.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'Please upload a ZIP file'}), 400

    # Extract ZIP to a temp directory
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, 'upload.zip')
    file.save(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp_dir)
    except Exception as e:
        return jsonify({'error': f'Could not open ZIP: {str(e)}'}), 400

    # Find all image files
    image_files = []
    for root, dirs, files in os.walk(tmp_dir):
        # Skip macOS metadata folders
        dirs[:] = [d for d in dirs if d != '__MACOSX']
        for f in files:
            if Path(f).suffix.lower() in ALLOWED_EXTENSIONS:
                image_files.append(os.path.join(root, f))

    if len(image_files) == 0:
        return jsonify({'error': 'No images found in ZIP file'}), 400

    if len(image_files) > 100:
        return jsonify({'error': 'Maximum 100 images per upload'}), 400

    # Score each image visually
    cows = []
    for i, img_path in enumerate(sorted(image_files)):
        try:
            low_ch4_prob, confidence = score_image(img_path)
            cows.append({
                'id': i,
                'filename': Path(img_path).name,
                'visual_score': round(low_ch4_prob * 100, 1),
                'confidence': round(confidence * 100, 1),
                'final_score': round(low_ch4_prob * 100, 1),
                'has_records': False,
                'tmp_path': img_path,
            })
        except Exception as e:
            cows.append({
                'id': i,
                'filename': Path(img_path).name,
                'error': str(e),
                'visual_score': 50.0,
                'confidence': 0.0,
                'final_score': 50.0,
                'has_records': False,
                'tmp_path': img_path,
            })

    # Store tmp_dir in session via a simple file-based approach
    session_id = str(abs(hash(zip_path)))
    session_file = Path(tmp_dir) / 'session_id.txt'
    session_file.write_text(session_id)

    # Store cows data (without tmp_path for JSON response)
    response_cows = [{k: v for k, v in cow.items() if k != 'tmp_path'}
                     for cow in cows]

    return jsonify({
        'session_id': session_id,
        'tmp_dir': tmp_dir,
        'cows': response_cows,
        'total': len(cows)
    })


@app.route('/api/update_records', methods=['POST'])
def update_records():
    """
    Step 2: receive farm records for one or more cows,
    re-score them using both visual + formula streams.
    """
    data = request.json
    tmp_dir = data.get('tmp_dir')
    cow_records = data.get('records', [])

    if not tmp_dir or not os.path.exists(tmp_dir):
        return jsonify({'error': 'Session expired. Please re-upload.'}), 400

    results = []
    for record in cow_records:
        cow_id = record['id']
        filename = record['filename']
        farm_data = record.get('farm_data', {})

        img_path = None
        for root, dirs, files in os.walk(tmp_dir):
            dirs[:] = [d for d in dirs if d != '__MACOSX']
            for f in files:
                if f == filename:
                    img_path = os.path.join(root, f)
                    break

        if img_path is None:
            results.append({'id': cow_id, 'error': 'Image not found'})
            continue

        try:
            # Build formula vector from provided farm records
            formula_vec = get_formula_vector(
                DMI=farm_data.get('DMI'),
                GEI=farm_data.get('GEI'),
                MEI=farm_data.get('MEI'),
                NDF=farm_data.get('NDF'),
                NDFI=farm_data.get('NDFI'),
                NDFP=farm_data.get('NDFP'),
                FP=farm_data.get('FP'),
                ECMY=farm_data.get('ECMY'),
                LWT=farm_data.get('LWT'),
            )

            low_ch4_prob, confidence = score_image(img_path, formula_vec)
            results.append({
                'id': cow_id,
                'final_score': round(low_ch4_prob * 100, 1),
                'confidence': round(confidence * 100, 1),
                'has_records': True,
            })
        except Exception as e:
            results.append({'id': cow_id, 'error': str(e)})

    return jsonify({'updated': results})


@app.route('/api/breeding_pairs', methods=['POST'])
def breeding_pairs():
    """
    Step 3: given final scores for all cows,
    compute the best breeding pairs.
    """
    data = request.json
    cows = data.get('cows', [])
    top_n = data.get('top_n', 10)

    if len(cows) < 2:
        return jsonify({'error': 'Need at least 2 cows to compute pairs'}), 400

    import itertools
    pairs = []
    for c1, c2 in itertools.combinations(cows, 2):
        s1 = c1['final_score'] / 100
        s2 = c2['final_score'] / 100
        mean_score, std_score = predict_offspring(s1, s2)
        pairs.append({
            'cow1': {'id': c1['id'], 'filename': c1['filename'], 'score': c1['final_score']},
            'cow2': {'id': c2['id'], 'filename': c2['filename'], 'score': c2['final_score']},
            'offspring_score': round(mean_score * 100, 1),
            'offspring_std': round(std_score * 100, 1),
        })

    pairs.sort(key=lambda x: x['offspring_score'], reverse=True)

    return jsonify({
        'pairs': pairs[:top_n],
        'total_pairs_evaluated': len(pairs)
    })



@app.route('/api/report', methods=['POST'])
def generate_report():
    data = request.json
    cows = data.get('cows', [])
    pairs = data.get('pairs', [])
    buffer = io.BytesIO()
    GREEN = colors.HexColor('#1a5c2e')
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('t', fontSize=24, textColor=GREEN, fontName='Helvetica-Bold', spaceAfter=2, leading=28)
    sub_style = ParagraphStyle('s', fontSize=11, textColor=colors.HexColor('#666666'), fontName='Helvetica', spaceBefore=6, spaceAfter=20, leading=14)
    heading_style = ParagraphStyle('h', fontSize=14, textColor=GREEN, fontName='Helvetica-Bold', spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle('b', fontSize=10, fontName='Helvetica', textColor=colors.HexColor('#1a1a1a'), spaceAfter=6)
    disc_style = ParagraphStyle('d', fontSize=9, fontName='Helvetica', textColor=colors.HexColor('#5a4010'), backColor=colors.HexColor('#fff3dc'), borderPadding=8, spaceAfter=12)
    story = []
    story.append(Paragraph('Green Cattle', title_style))
    story.append(Paragraph('CH4 Methane Risk Analysis and Breeding Recommendations', sub_style))
    story.append(Paragraph(f'Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}', body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=GREEN, spaceAfter=16))
    story.append(Paragraph('Herd Summary', heading_style))
    dams = [c for c in cows if c.get("sex") == "dam"]
    sires = [c for c in cows if c.get("sex") == "sire"]
    avg = sum(c["final_score"] for c in cows) / len(cows) if cows else 0
    summary = [["Total cows", str(len(cows))], ["Dams", str(len(dams))], ["Sires", str(len(sires))], ["Avg CH4 risk", f"{avg:.1f}%"], ["Pairs evaluated", str(len(pairs))]]
    t = Table(summary, colWidths=[9*cm, 7*cm])
    t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),10),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,colors.HexColor("#f5f5f3")]),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e0e0dc")),("PADDING",(0,0),(-1,-1),6)]))
    story.append(t)
    story.append(Spacer(1,16))
    story.append(Paragraph('Individual Cow Scores', heading_style))
    sorted_cows = sorted(cows, key=lambda c: c["final_score"], reverse=True)
    cow_data = [["Rank","Cow","Sex","CH4 Risk","Source","Confidence"]]
    for i, cow in enumerate(sorted_cows):
        sex = "Dam" if cow.get("sex")=="dam" else "Sire" if cow.get("sex")=="sire" else "?"
        risk = "Low" if cow["final_score"]>=60 else "Medium" if cow["final_score"]>=40 else "High"
        src = "Visual+records" if cow.get("has_records") else "Visual only"
        cow_data.append([str(i+1), cow["filename"], sex, f"{cow['final_score']}% {risk}", src, f"{cow['confidence']}%"])
    ct = Table(cow_data, colWidths=[1.2*cm,5.5*cm,2*cm,3.5*cm,3.5*cm,2.3*cm])
    ct.setStyle(TableStyle([("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),("BACKGROUND",(0,0),(-1,0),GREEN),("TEXTCOLOR",(0,0),(-1,0),colors.white),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5f3")]),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e0e0dc")),("PADDING",(0,0),(-1,-1),5)]))
    story.append(ct)
    story.append(Spacer(1,16))
    story.append(Paragraph('Breeding Recommendations', heading_style))
    story.append(Paragraph('All valid Dam x Sire pairs ranked by predicted offspring methane risk.', body_style))
    if pairs:
        pair_data = [["Rank","Dam","Dam score","Sire","Sire score","Offspring","+-"]]
        for i, pair in enumerate(pairs[:50]):
            c1 = next((c for c in cows if c["id"]==pair["cow1"]["id"]),None)
            dam = pair["cow1"] if c1 and c1.get("sex")=="dam" else pair["cow2"]
            sire = pair["cow1"] if c1 and c1.get("sex")=="sire" else pair["cow2"]
            pair_data.append([f"#{i+1}", dam["filename"], f"{dam['score']}%", sire["filename"], f"{sire['score']}%", f"{pair['offspring_score']}%", f"+-{pair['offspring_std']}%"])
        pt = Table(pair_data, colWidths=[1*cm,3.5*cm,2*cm,3.5*cm,2*cm,2.5*cm,1.5*cm])
        pt.setStyle(TableStyle([("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),8),("BACKGROUND",(0,0),(-1,0),GREEN),("TEXTCOLOR",(0,0),(-1,0),colors.white),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5f3")]),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e0e0dc")),("PADDING",(0,0),(-1,-1),4)]))
        story.append(pt)
    story.append(Spacer(1,20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0dc"), spaceAfter=12))
    story.append(Paragraph("<b>Disclaimer:</b> These recommendations are based on breed-typical proxy labels and a simplified additive genetic model. They do not reflect real individual methane measurements. For real on-farm breeding decisions, consult a livestock genetics specialist.", disc_style))
    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="green_cattle_report.pdf")


if __name__ == '__main__':
    app.run(debug=True, port=8080)