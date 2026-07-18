const state = { file: null, tmpDir: null, sessionId: null, cows: [], currentCowIndex: 0, farmRecords: {}, validPairs: [] }

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'))
  document.getElementById(id).classList.add('active')
  window.scrollTo(0, 0)
}

const dropZone = document.getElementById('drop-zone')
const fileInput = document.getElementById('file-input')
const uploadBtn = document.getElementById('upload-btn')
const fileNameEl = document.getElementById('file-name')

fileInput.addEventListener('change', e => { if (e.target.files[0]) setFile(e.target.files[0]) })
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over') })
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'))
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over')
  const f = e.dataTransfer.files[0]
  if (f && f.name.endsWith('.zip')) setFile(f)
})

function setFile(f) {
  state.file = f
  fileNameEl.textContent = f.name + ' (' + (f.size/1024/1024).toFixed(1) + ' MB)'
  uploadBtn.disabled = false
}

uploadBtn.addEventListener('click', async () => {
  if (!state.file) return
  showScreen('screen-analyzing')
  setProgress(10, 'Uploading images...')
  const formData = new FormData()
  formData.append('file', state.file)
  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData })
    const data = await res.json()
    if (data.error) { alert('Error: ' + data.error); showScreen('screen-upload'); return }
    setProgress(100, 'Analysis complete!')
    state.cows = data.cows.map(c => ({ ...c, sex: 'unknown' }))
    state.tmpDir = data.tmp_dir
    state.farmRecords = {}
    state.currentCowIndex = 0
    setTimeout(() => { buildSexScreen(); showScreen('screen-sex') }, 500)
  } catch (err) { alert('Upload failed: ' + err.message); showScreen('screen-upload') }
})

function setProgress(pct, msg) {
  document.getElementById('progress-fill').style.width = pct + '%'
  document.getElementById('analyzing-sub').textContent = msg
}

function buildSexScreen() {
  const grid = document.getElementById('sex-grid')
  grid.className = 'sex-grid'
  grid.innerHTML = state.cows.map((cow, i) => `
    <div class="sex-card">
      <div class="cow-filename">${cow.filename}</div>
      <div class="cow-visual-score">${cow.visual_score}% low-CH4</div>
      <div class="sex-toggle">
        <button onclick="setSex(${i},'dam')" id="btn-dam-${i}">Dam ♀</button>
        <button onclick="setSex(${i},'sire')" id="btn-sire-${i}">Sire ♂</button>
        <button onclick="setSex(${i},'unknown')" id="btn-unk-${i}">?</button>
      </div>
    </div>`).join('')
  const half = Math.ceil(state.cows.length / 2)
  state.cows.forEach((_, i) => setSex(i, i < half ? 'dam' : 'sire'))
}

function setSex(idx, sex) {
  state.cows[idx].sex = sex
  document.getElementById(`btn-dam-${idx}`).className = sex === 'dam' ? 'active-dam' : ''
  document.getElementById(`btn-sire-${idx}`).className = sex === 'sire' ? 'active-sire' : ''
  document.getElementById(`btn-unk-${idx}`).className = sex === 'unknown' ? 'active-unknown' : ''
}

document.getElementById('confirm-sex').addEventListener('click', () => {
  state.currentCowIndex = 0
  buildRecordsScreen()
  showScreen('screen-records')
})

const fields = [
  { key: 'DMI', label: 'Dry Matter Intake', unit: 'kg/day', hint: 'How much dry feed the cow eats' },
  { key: 'GEI', label: 'Gross Energy Intake', unit: 'MJ/day', hint: 'Energy content of the feed' },
  { key: 'MEI', label: 'Metabolisable Energy', unit: 'MJ/day', hint: 'Usable energy from feed' },
  { key: 'LWT', label: 'Live Weight', unit: 'kg', hint: 'Body weight of the cow' },
  { key: 'ECMY', label: 'Milk Yield (ECM)', unit: 'kg/day', hint: 'Energy corrected milk yield' },
  { key: 'NDF', label: 'NDF Content', unit: 'g/kg DMI', hint: 'Neutral detergent fiber in feed' },
  { key: 'NDFI', label: 'NDF Intake', unit: 'kg/day', hint: 'Actual kg of fiber eaten' },
  { key: 'NDFP', label: 'NDF Proportion', unit: '%', hint: 'Percentage of diet that is fiber' },
  { key: 'FP', label: 'Forage Proportion', unit: '0-1', hint: 'Fraction of diet that is forage' },
]

function buildRecordsScreen() { renderCowForm(state.currentCowIndex); updateRecordsNav() }

function renderCowForm(idx) {
  const cow = state.cows[idx]
  const saved = state.farmRecords[cow.id] || {}
  const sexLabel = cow.sex === 'dam' ? ' ♀ Dam' : cow.sex === 'sire' ? ' ♂ Sire' : ''
  document.getElementById('cow-counter').textContent = `Cow ${idx+1} of ${state.cows.length}`
  document.getElementById('cow-record-form').innerHTML = `
    <div class="cow-form-header">
      <div>
        <div class="cow-name">${cow.filename}${sexLabel}</div>
        <div class="muted">Visual score: ${cow.visual_score}% low-CH4 probability</div>
      </div>
    </div>
    <div class="form-grid">
      ${fields.map(f => `
        <div class="form-group">
          <label>${f.label} <span style="font-weight:400">(${f.unit})</span></label>
          <input type="number" step="any" placeholder="optional" data-key="${f.key}"
            value="${saved[f.key] !== undefined ? saved[f.key] : ''}">
          <div class="form-hint">${f.hint}</div>
        </div>`).join('')}
    </div>`
}

function saveCurrentCowRecords() {
  const cow = state.cows[state.currentCowIndex]
  const inputs = document.querySelectorAll('#cow-record-form input[data-key]')
  const records = {}
  inputs.forEach(inp => { if (inp.value !== '') records[inp.dataset.key] = parseFloat(inp.value) })
  if (Object.keys(records).length > 0) state.farmRecords[cow.id] = records
}

function updateRecordsNav() {
  document.getElementById('prev-cow').disabled = state.currentCowIndex === 0
  document.getElementById('next-cow').textContent = state.currentCowIndex === state.cows.length - 1 ? 'Done ✓' : 'Next →'
}

document.getElementById('prev-cow').addEventListener('click', () => {
  saveCurrentCowRecords(); state.currentCowIndex--; renderCowForm(state.currentCowIndex); updateRecordsNav()
})
document.getElementById('next-cow').addEventListener('click', () => {
  saveCurrentCowRecords()
  if (state.currentCowIndex < state.cows.length - 1) { state.currentCowIndex++; renderCowForm(state.currentCowIndex); updateRecordsNav() }
  else finishRecords()
})
document.getElementById('skip-all-records').addEventListener('click', () => { state.farmRecords = {}; finishRecords() })
document.getElementById('finish-records').addEventListener('click', () => { saveCurrentCowRecords(); finishRecords() })

async function finishRecords() {
  showScreen('screen-analyzing')
  setProgress(20, 'Updating scores with farm records...')
  const recordsToUpdate = Object.entries(state.farmRecords).map(([id, farm_data]) => {
    const cow = state.cows.find(c => c.id === parseInt(id))
    return { id: parseInt(id), filename: cow.filename, farm_data }
  })
  if (recordsToUpdate.length > 0) {
    try {
      const res = await fetch('/api/update_records', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tmp_dir: state.tmpDir, records: recordsToUpdate }) })
      const data = await res.json()
      data.updated.forEach(u => {
        const cow = state.cows.find(c => c.id === u.id)
        if (cow && !u.error) { cow.final_score = u.final_score; cow.confidence = u.confidence; cow.has_records = true }
      })
    } catch (err) { console.error('Record update failed:', err) }
  }
  setProgress(60, 'Computing breeding recommendations...')
  try {
    const res = await fetch('/api/breeding_pairs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cows: state.cows, top_n: 999 }) })
    const data = await res.json()
    setProgress(100, 'Done!')
    setTimeout(() => { buildResultsScreen(data.pairs, data.total_pairs_evaluated); showScreen('screen-results') }, 400)
  } catch (err) { alert('Failed: ' + err.message); showScreen('screen-upload') }
}

function scoreClass(s) { return s >= 60 ? 'score-low' : s >= 40 ? 'score-mid' : 'score-high' }
function scoreLabel(s) { return s >= 60 ? 'Low risk' : s >= 40 ? 'Medium risk' : 'High risk' }

function buildResultsScreen(pairs, total) {
  buildHerdTable()
  const validPairs = filterValidPairs(pairs)
  state.validPairs = validPairs
  buildPairsTable(validPairs, total)
  buildBreedingTreeSVG(validPairs.slice(0, 5))
}

function filterValidPairs(pairs) {
  return pairs.filter(p => {
    const c1 = state.cows.find(c => c.id === p.cow1.id)
    const c2 = state.cows.find(c => c.id === p.cow2.id)
    if (!c1 || !c2) return true
    return (c1.sex === 'dam' && c2.sex === 'sire') || (c1.sex === 'sire' && c2.sex === 'dam') || c1.sex === 'unknown' || c2.sex === 'unknown'
  })
}

function getDamSire(pair) {
  const c1 = state.cows.find(c => c.id === pair.cow1.id)
  const c2 = state.cows.find(c => c.id === pair.cow2.id)
  const dam = c1?.sex === 'dam' ? pair.cow1 : c2?.sex === 'dam' ? pair.cow2 : pair.cow1
  const sire = c1?.sex === 'sire' ? pair.cow1 : c2?.sex === 'sire' ? pair.cow2 : pair.cow2
  return { dam, sire }
}

function buildHerdTable() {
  const sorted = [...state.cows].sort((a, b) => b.final_score - a.final_score)
  document.getElementById('herd-tbody').innerHTML = sorted.map((cow, i) => `
    <tr>
      <td>${i+1}</td>
      <td><strong>${cow.filename}</strong></td>
      <td>${cow.sex === 'dam' ? '♀ Dam' : cow.sex === 'sire' ? '♂ Sire' : '?'}</td>
      <td><span class="score-pill ${scoreClass(cow.final_score)}">${cow.final_score}% — ${scoreLabel(cow.final_score)}</span></td>
      <td><span class="data-badge ${cow.has_records ? 'data-multi' : 'data-visual'}">${cow.has_records ? 'Visual + records' : 'Visual only'}</span></td>
      <td>${cow.confidence}%</td>
    </tr>`).join('')
}

function buildPairsTable(pairs, total) {
  document.getElementById('pairs-tbody').innerHTML = pairs.map((pair, i) => {
    const { dam, sire } = getDamSire(pair)
    return `<tr>
      <td><strong>#${i+1}</strong></td>
      <td>${dam.filename}<br><small class="muted">♀ ${dam.score}% low-risk</small></td>
      <td>${sire.filename}<br><small class="muted">♂ ${sire.score}% low-risk</small></td>
      <td><span class="score-pill ${scoreClass(pair.offspring_score)}">${pair.offspring_score}%</span></td>
      <td><small class="muted">±${pair.offspring_std}%</small></td>
    </tr>`
  }).join('')
  const note = document.createElement('p')
  note.className = 'muted'
  note.style.marginTop = '12px'
  note.textContent = `${pairs.length} valid Dam x Sire pairs shown (${total} total evaluated)`
  document.getElementById('pairs-table').parentNode.appendChild(note)
}

function buildBreedingTreeSVG(pairs) {
  if (!pairs.length) return
  const W = 800, rowH = 110, H = 60 + pairs.length * rowH + 40
  let svg = `<svg width="100%" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="font-family:-apple-system,sans-serif">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#999" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
  <text x="20" y="30" font-size="13" font-weight="600" fill="#1a1a1a">Top ${pairs.length} Dam x Sire pairings</text>`
  pairs.forEach((pair, i) => {
    const y = 50 + i * rowH
    const { dam, sire } = getDamSire(pair)
    const rank = i + 1
    const rankColor = rank === 1 ? '#1a5c2e' : rank === 2 ? '#3b6d11' : '#666'
    const boxStyle = (score) => ({ fill: score >= 60 ? '#dff0e5' : score >= 40 ? '#fff3dc' : '#fbe3e3', border: score >= 60 ? '#3b6d11' : score >= 40 ? '#ba7517' : '#a32d2d' })
    const d = boxStyle(dam.score), s = boxStyle(sire.score), o = boxStyle(pair.offspring_score)
    const oText = pair.offspring_score >= 60 ? '#1a5c2e' : pair.offspring_score >= 40 ? '#7a4a00' : '#7a1a1a'
    svg += `
    <circle cx="26" cy="${y+32}" r="18" fill="${rankColor}"/>
    <text x="26" y="${y+37}" text-anchor="middle" font-size="13" font-weight="600" fill="white">#${rank}</text>
    <rect x="56" y="${y+8}" width="200" height="52" rx="8" fill="${d.fill}" stroke="${d.border}" stroke-width="1"/>
    <text x="156" y="${y+24}" text-anchor="middle" font-size="11" fill="#666">♀ Dam</text>
    <text x="156" y="${y+40}" text-anchor="middle" font-size="11" font-weight="600" fill="#1a1a1a">${dam.filename}</text>
    <text x="156" y="${y+54}" text-anchor="middle" font-size="11" fill="#666">${dam.score}% low-risk</text>
    <text x="270" y="${y+38}" text-anchor="middle" font-size="18" fill="#999">x</text>
    <rect x="284" y="${y+8}" width="200" height="52" rx="8" fill="${s.fill}" stroke="${s.border}" stroke-width="1"/>
    <text x="384" y="${y+24}" text-anchor="middle" font-size="11" fill="#666">♂ Sire</text>
    <text x="384" y="${y+40}" text-anchor="middle" font-size="11" font-weight="600" fill="#1a1a1a">${sire.filename}</text>
    <text x="384" y="${y+54}" text-anchor="middle" font-size="11" fill="#666">${sire.score}% low-risk</text>
    <line x1="496" y1="${y+34}" x2="530" y2="${y+34}" stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>
    <rect x="534" y="${y+8}" width="220" height="52" rx="8" fill="${o.fill}" stroke="${o.border}" stroke-width="1.5"/>
    <text x="644" y="${y+24}" text-anchor="middle" font-size="11" fill="#666">Predicted offspring</text>
    <text x="644" y="${y+46}" text-anchor="middle" font-size="15" font-weight="700" fill="${oText}">${pair.offspring_score}% low-risk</text>`
  })
  svg += `</svg>`
  document.getElementById('breeding-tree-svg').innerHTML = svg
}

document.getElementById('download-report').addEventListener('click', async () => {
  const btn = document.getElementById('download-report')
  btn.textContent = 'Generating...'
  btn.disabled = true
  try {
    const res = await fetch('/api/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cows: state.cows, pairs: state.validPairs }) })
    if (!res.ok) throw new Error('Report generation failed')
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'green_cattle_report.pdf'; a.click()
    URL.revokeObjectURL(url)
  } catch (err) { alert('Could not generate report: ' + err.message) }
  btn.textContent = '⬇ Download report'
  btn.disabled = false
})

document.getElementById('start-over').addEventListener('click', () => {
  state.file = null; state.tmpDir = null; state.cows = []; state.farmRecords = {}
  document.getElementById('file-name').textContent = ''
  uploadBtn.disabled = true
  showScreen('screen-upload')
})
