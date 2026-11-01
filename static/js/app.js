(() => {
  'use strict';
  const body = document.body;
  // UI polish only: keep the browser title aligned with the current product name.
  document.title = 'Malware Detector | Security Scanner';
  const MAX_MB = Number(body.dataset.maxUploadMb);
  const MAX_URL = Number(body.dataset.maxUrlLength);

  // ---------- helpers (all dynamic text is written with textContent, so it is never parsed as markup)
  const $ = (id) => document.getElementById(id);
  const el = (tag, props = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === 'class') n.className = v; else if (k === 'text') n.textContent = v; else n.setAttribute(k, v);
    }
    for (const k of kids) if (k != null) n.append(k);
    return n;
  };
  const svg = (tag, attrs = {}) => {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const pct = (x) => (x * 100).toFixed(1) + '%';

  async function api(path, options) {
    let res;
    try { res = await fetch(path, options); }
    catch { throw new Error('Could not reach the server. Check that the app is still running.'); }
    let payload = null;
    try { payload = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok || !payload || payload.ok !== true) {
      throw new Error((payload && payload.error && payload.error.message) || `Request failed (HTTP ${res.status}).`);
    }
    return payload.data;
  }

  function setBusy(form, resultBox, busy) {
    const btn = form.querySelector('button.primary');
    btn.disabled = busy; btn.classList.toggle('busy', busy);
    btn.querySelector('.btn-label').textContent = busy ? 'Working…' : btn.dataset.label;
    resultBox.setAttribute('aria-busy', String(busy));
  }
  document.querySelectorAll('button.primary').forEach((b) => { b.dataset.label = b.querySelector('.btn-label').textContent; });

  function showError(box, message) {
    box.replaceChildren(el('div', { class: 'card error-card' }, el('h2', { text: 'That did not work' }), el('p', { text: message })));
  }

  // ---------- tabs
  const tabs = [$('tab-url'), $('tab-file')];
  function selectTab(tab) {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on)); t.tabIndex = on ? 0 : -1;
      $(t.getAttribute('aria-controls')).hidden = !on;
    });
    tab.focus();
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => selectTab(t));
    t.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); selectTab(tabs[(i + 1) % tabs.length]); }
    });
  });

  // ---------- gauge shared by URL result
  function gauge(score, threshold) {
    const track = el('div', { class: 'track', role: 'img', 'aria-label': `Score ${score.toFixed(2)} on a scale from 0 to 1; threshold ${threshold}` });
    track.append(el('span', { class: 'tick', style: `left:${threshold * 100}%` }));
    track.append(el('span', { class: 'tick-label', style: `left:${threshold * 100}%`, text: `threshold ${threshold}` }));
    track.append(el('span', { class: 'pin', style: `left:${Math.min(Math.max(score, 0), 1) * 100}%` }));
    return el('div', { class: 'gauge' }, track, el('div', { class: 'scale' }, el('span', { text: '0' }), el('span', { text: '1' })));
  }

  // ---------- URL
  const urlForm = $('url-form'), urlInput = $('url-input'), urlError = $('url-error'), urlResult = $('url-result');
  function fieldError(node, msg) { node.hidden = !msg; node.textContent = msg || ''; }

  function renderUrl(d) {
    const mal = d.label === 'malicious';
    const card = el('div', { class: 'card' },
      el('div', { class: `verdict ${d.label}` }, el('h2', { text: d.prediction }), el('span', { class: 'badge', text: `${d.risk_band} risk` })),
      el('p', { class: 'analysed', text: `Analysed as: ${d.normalized_url}` }),
      gauge(d.malicious_score, d.threshold),
      el('p', { class: 'score-line' }, 'Malicious score ', el('b', { text: d.malicious_score.toFixed(4) })),
      el('p', { class: 'explain', text: 'The score is the model\u2019s estimated chance that the URL text resembles malicious URLs in its training data. ' +
        'It is not proof either way: a high score does not confirm an attack and a low score does not guarantee safety.' }));
    if (d.bare_domain) card.append(el('p', { class: 'note', text: 'Only a domain was entered. Scores are more informative for full links that include a path.' }));
    if (d.signals.length && d.risk_band !== 'low') {
      card.append(el('h3', { class: 'sub', text: 'Text patterns pushing the score up' }));
      card.append(el('ul', { class: 'signals' }, ...d.signals.map((s) => el('li', { text: s.pattern.replace(/ /g, '\u2423') }))));
    }
    urlResult.replaceChildren(card);
  }

  urlForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const value = urlInput.value.trim();
    fieldError(urlError, '');
    if (!value) { fieldError(urlError, 'Enter a URL to analyse.'); urlInput.focus(); return; }
    if (value.length > MAX_URL) { fieldError(urlError, `The URL is longer than ${MAX_URL} characters.`); return; }
    if (/\s/.test(value)) { fieldError(urlError, 'The URL contains spaces. Remove them and try again.'); return; }
    setBusy(urlForm, urlResult, true);
    try {
      renderUrl(await api('/api/predict-url', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url: value }) }));
    } catch (err) { showError(urlResult, err.message); }
    finally { setBusy(urlForm, urlResult, false); }
  });

  // ---------- CSV
  const fileForm = $('file-form'), fileInput = $('file-input'), fileError = $('file-error'), fileResult = $('file-result'), drop = $('drop'), dropTitle = $('drop-title');
  const DROP_DEFAULT = dropTitle.textContent;

  function validateFile(f) {
    if (!f) return 'Choose a CSV file first.';
    if (!/\.csv$/i.test(f.name)) return 'Only .csv files are accepted.';
    if (f.size === 0) return 'The file is empty.';
    if (f.size > MAX_MB * 1024 * 1024) return `The file is larger than ${MAX_MB} MB.`;
    return '';
  }
  function chosen() {
    const f = fileInput.files[0];
    dropTitle.textContent = f ? `${f.name} (${(f.size / 1024).toFixed(1)} KB)` : DROP_DEFAULT;
    fieldError(fileError, f ? validateFile(f) : '');
  }
  fileInput.addEventListener('change', chosen);
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; chosen(); }
  });

  function histogram(counts, threshold) {
    const W = 300, H = 100, max = Math.max(...counts, 1), bw = W / counts.length;
    const s = svg('svg', { class: 'hist', viewBox: `0 0 ${W} ${H + 16}`, role: 'img', 'aria-label': 'Distribution of malware scores across rows' });
    counts.forEach((c, i) => {
      const h = (c / max) * H;
      s.append(svg('rect', { x: i * bw + 2, y: H - h, width: bw - 4, height: h, rx: 2 }));
    });
    s.append(svg('line', { x1: threshold * W, x2: threshold * W, y1: 0, y2: H }));
    ['0', '0.5', '1'].forEach((t, i) => { const x = svg('text', { x: i * W / 2, y: H + 13, 'text-anchor': i === 0 ? 'start' : i === 2 ? 'end' : 'middle' }); x.textContent = t; s.append(x); });
    return s;
  }

  function renderFile(d) {
    const s = d.summary;
    const stat = (cls, label, value) => el('div', { class: `stat ${cls}` }, el('dt', { text: label }), el('dd', { text: value }));
    const card = el('div', { class: 'card' },
      el('div', { class: 'verdict' }, el('h2', { text: `${s.malware.toLocaleString()} of ${s.rows.toLocaleString()} rows flagged as malware` })),
      el('dl', { class: 'stats' },
        stat('', 'Rows scanned', s.rows.toLocaleString()), stat('mal', 'Malware', `${s.malware.toLocaleString()} (${s.malware_pct}%)`),
        stat('ben', 'Benign', s.benign.toLocaleString()), stat('', 'Mean score', s.mean_score.toFixed(3))),
      el('div', { class: 'split', role: 'img', 'aria-label': `${s.malware_pct}% malware, ${(100 - s.malware_pct).toFixed(2)}% benign` },
        el('span', { class: 'm', style: `width:${s.malware_pct}%` }), el('span', { class: 'b', style: `width:${100 - s.malware_pct}%` })),
      el('h3', { class: 'sub', text: 'Score distribution (dashed line = threshold)' }), histogram(s.score_histogram, s.threshold),
      el('p', { class: 'explain', text: `Whole-file reading: mean score ${s.mean_score.toFixed(3)} \u2192 ${s.file_level_label}. ${s.file_level_note}` }),
      el('p', { class: 'explain', text: 'Scores are model outputs, not verified labels. Each row is judged on its own 27 numbers; a high score is not proof of malware.' }));
    d.warnings.forEach((w) => card.append(el('p', { class: 'note', text: w })));
    const link = el('a', { class: 'dl', href: `${d.download.url}?name=${encodeURIComponent(d.download.filename)}`, download: d.download.filename, text: 'Download predictions (CSV)' });
    card.append(link);
    card.append(el('h3', { class: 'sub', text: `First ${d.preview.length} rows` }));
    const table = el('table', {}, el('thead', {}, el('tr', {}, ...['CSV row', 'Score', 'Prediction'].map((h) => el('th', { text: h })))),
      el('tbody', {}, ...d.preview.map((r) => el('tr', {}, el('td', { text: String(r.row) }), el('td', { text: r.malware_score.toFixed(4) }),
        el('td', { class: r.prediction === 'Malware' ? 'm' : 'b', text: r.prediction })))));
    card.append(el('div', { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': 'Row predictions' }, table));
    fileResult.replaceChildren(card);
  }

  fileForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = fileInput.files[0], problem = validateFile(f);
    if (problem) { fieldError(fileError, problem); return; }
    fieldError(fileError, '');
    setBusy(fileForm, fileResult, true);
    try {
      const fd = new FormData(); fd.append('file', f);
      renderFile(await api('/api/predict-file', { method: 'POST', body: fd }));
    } catch (err) { showError(fileResult, err.message); }
    finally { setBusy(fileForm, fileResult, false); }
  });

  // ---------- status banner + model details
  (async () => {
    try {
      const h = await api('/api/health');
      const problems = [];
      if (!h.url_model.loaded) problems.push('URL model: ' + h.url_model.error);
      if (!h.file_model.loaded) problems.push('CSV model: ' + h.file_model.error);
      if (problems.length) { const b = $('status-banner'); b.hidden = false; b.textContent = problems.join(' | '); }
    } catch { /* the banner is optional */ }
    try {
      const info = await api('/api/model-info');
      const box = $('model-info'); box.replaceChildren();
      if (info.url) {
        const m = info.url.held_out_metrics;
        box.append(el('div', {}, el('h3', { text: 'URL model' }), el('p', { text: info.url.algorithm }),
          el('ul', {}, el('li', { text: `Held-out domains: accuracy ${pct(m.accuracy)}, malicious recall ${pct(m.recall_bad)}, malicious precision ${pct(m.precision_bad)}` }),
            el('li', { text: 'Trained on text patterns only. It can flag legitimate sites whose names resemble phishing targets.' }))));
      }
      if (info.file) {
        const m = info.file.grouped_cv_metrics;
        box.append(el('div', {}, el('h3', { text: 'CSV malware model' }), el('p', { text: info.file.algorithm }),
          el('ul', {}, el('li', { text: `Unseen apps (grouped cross-validation): accuracy ${pct(m.accuracy)}, malware recall ${pct(m.recall_malware)}, precision ${pct(m.precision_malware)}` }),
            el('li', { text: 'Roughly 1 in 9 rows is misjudged even on apps the model has not seen; treat results as a screening aid.' }))));
      }
      if (!box.children.length) box.append(el('p', { text: 'No models are loaded.' }));
    } catch { /* ignore */ }
  })();
})();
