/* Crowd Read: page logic. Loads the exported model, scores comments, and draws the pulse board. */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (x, d = 0) => (x * 100).toFixed(d) + '%';
  const int = n => Number(n).toLocaleString('en-US');

  // Topics people mention, found by keyword in each clause
  const ASPECTS = [
    ['Acting', 'cast and performances', ['acting', 'actor', 'actors', 'actress', 'cast', 'performance', 'performances', 'performer', 'role', 'roles']],
    ['Story', 'plot and script', ['story', 'plot', 'script', 'writing', 'twist', 'twists', 'premise', 'storyline']],
    ['Visuals', 'look and effects', ['visuals', 'visually', 'cinematography', 'shot', 'shots', 'cgi', 'effects', 'look', 'looks', 'gorgeous', 'stunning']],
    ['Pacing', 'length and speed', ['pacing', 'paced', 'slow', 'long', 'length', 'drag', 'drags', 'dragged', 'runtime', 'hour', 'hours', 'minutes']],
    ['Music', 'score and soundtrack', ['music', 'soundtrack', 'score', 'song', 'songs']],
    ['Ending', 'finale', ['ending', 'ended', 'finale', 'end']],
    ['Dialogue', 'lines and humor', ['dialogue', 'lines', 'jokes', 'joke', 'funny']],
    ['Characters', 'who is on screen', ['character', 'characters', 'villain', 'hero', 'kids']],
  ];

  const HARNESS = [ // from NLP_Algo_Harness.ipynb, test set of 10,000 reviews
    ['Logistic Regression', 0.8849, 0.8872], ['CatBoost', 0.8740, 0.8775], ['LightGBM', 0.8599, 0.8629], ['XGBoost', 0.8572, 0.8614],
    ['Random Forest', 0.8508, 0.8500], ['Multinomial NB', 0.8490, 0.8508], ['Gradient Boosting', 0.8122, 0.8230], ['AdaBoost', 0.7494, 0.7575],
  ];

  let model = null, sample = null, source = 'sample', fileComments = null, fileRows = null;

  function clauses(text) {
    return String(text).split(/(?<=[.!?;])\s+|\s+(?:but|though|although|however|yet)\s+/i).map(s => s.trim()).filter(s => s.length > 2);
  }
  function wordsOf(text) { return (String(text).toLowerCase().match(/[a-z]+/g) || []); }

  function analyze(comments) {
    const scored = comments.map((text, i) => ({ i, text, ...model.score(text) }));
    const asp = ASPECTS.map(([name, sub]) => ({ name, sub, n: 0, pos: 0 }));
    for (const c of scored) {
      for (const cl of clauses(c.text)) {
        const w = new Set(wordsOf(cl));
        const hits = ASPECTS.map(([, , keys], k) => keys.some(key => w.has(key)) ? k : -1).filter(k => k >= 0);
        if (!hits.length) continue;
        const p = model.score(cl).p;
        for (const k of hits) { asp[k].n++; if (p >= 0.5) asp[k].pos++; }
      }
    }
    return { scored, asp };
  }

  /* ---------- rendering ---------- */
  function renderHero(res) {
    const pos = res.scored.filter(c => c.p >= 0.5).length;
    $('#heroScore').textContent = Math.round(pos / res.scored.length * 100);
    $('#heroCount').textContent = int(res.scored.length);
    const top = res.asp.filter(a => a.n >= 3).sort((a, b) => b.n - a.n).slice(0, 4);
    $('#heroAspects').innerHTML = top.map(a => {
      const s = a.pos / a.n;
      return `<div class="ha"><span>${esc(a.name)}</span><span class="t"><i style="width:${s * 100}%;background:var(--pos)"></i><i style="width:${(1 - s) * 100}%;background:var(--neg)"></i></span><span class="v">${pct(s)}</span></div>`;
    }).join('');
  }

  function renderBoard(res) {
    const n = res.scored.length, pos = res.scored.filter(c => c.p >= 0.5).length;
    const sureNeg = res.scored.filter(c => c.p < 0.2).length, unsure = res.scored.filter(c => c.p >= 0.4 && c.p < 0.6).length;
    $('#tiles').innerHTML = [
      [int(n), 'comments read'], [pct(pos / n), 'audience score (share positive)', true],
      [int(sureNeg), 'clearly negative (under 20%)'], [int(unsure), 'hard to call (40–60%)'],
    ].map(([v, l, hl]) => `<div class="tile${hl ? ' hl' : ''}"><b>${v}</b><span>${l}</span></div>`).join('');

    const asp = res.asp.filter(a => a.n > 0).sort((a, b) => b.n - a.n);
    $('#aspects').innerHTML = asp.length ? asp.map(a => {
      const s = a.pos / a.n;
      return `<div class="asp"><span class="name">${esc(a.name)}<small>${esc(a.sub)}</small></span><span class="t" role="img" aria-label="${pct(s)} positive of ${a.n} mentions"><i style="width:${s * 100}%;background:var(--pos)"></i><i style="width:${(1 - s) * 100}%;background:var(--neg)"></i></span><span class="v">${pct(s)} · ${a.n}</span></div>`;
    }).join('') : '<p class="muted">None of the topic keywords appear in these comments.</p>';

    drawHist(res.scored);

    const item = c => `<li><button data-i="${c.i}" aria-pressed="false"><span class="pct ${c.p < 0.4 ? 'neg' : c.p >= 0.6 ? 'pos' : 'mid'}">${pct(c.p)}</span><span>${esc(c.text.length > 220 ? c.text.slice(0, 220) + '…' : c.text)}</span></button></li>`;
    const worst = res.scored.slice().sort((a, b) => a.p - b.p).slice(0, 6);
    const unsureList = res.scored.filter(c => c.p >= 0.35 && c.p < 0.65).sort((a, b) => Math.abs(a.p - 0.5) - Math.abs(b.p - 0.5)).slice(0, 6);
    $('#worst').innerHTML = worst.map(item).join('');
    $('#unsure').innerHTML = unsureList.length ? unsureList.map(item).join('') : '<li class="muted">Every comment was scored with some confidence.</li>';
    current = res;
    if (worst[0]) explain(worst[0].text, worst[0].i);
  }
  let current = null;

  function drawHist(scored) {
    const B = 10, bins = new Array(B).fill(0);
    scored.forEach(c => bins[Math.min(B - 1, Math.floor(c.p * B))]++);
    const W = 520, H = 210, m = { l: 34, r: 8, t: 10, b: 30 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const mx = Math.max(...bins, 1), step = Math.max(1, Math.ceil(mx / 4));
    let s = `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of comments by probability positive">`;
    for (let v = 0; v <= mx; v += step) { const y = m.t + ih - v / mx * ih; s += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/><text class="tick" x="${m.l - 6}" y="${y + 4}" text-anchor="end">${v}</text>`; }
    const bw = iw / B;
    bins.forEach((v, k) => {
      const h = v / mx * ih, x = m.l + k * bw + 2, col = k < 4 ? 'var(--neg)' : k >= 6 ? 'var(--pos)' : 'var(--mid)';
      s += `<rect x="${x.toFixed(1)}" y="${(m.t + ih - h).toFixed(1)}" width="${(bw - 4).toFixed(1)}" height="${h.toFixed(1)}" rx="3" style="fill:${col}"/>`;
    });
    for (let k = 0; k <= B; k += 2) s += `<text class="tick" x="${m.l + k * bw}" y="${H - 10}" text-anchor="middle">${k * 10}%</text>`;
    $('#hist').innerHTML = s + '</svg>';
  }

  function explain(text, i) {
    $$('.clist button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.i === i)));
    $('#oneBox').value = text;
    renderOne(text);
  }
  function renderOne(text) {
    const out = $('#oneOut');
    if (!text.trim()) { out.innerHTML = '<p class="muted">Type a comment to see how each word pushes the score.</p>'; return; }
    const r = model.score(text);
    const byWord = new Map(r.contrib.map(c => [c.word, c.value]));
    const mx = Math.max(...r.contrib.map(c => Math.abs(c.value)), 1e-9);
    const chips = r.processed.split(' ').filter(Boolean).map(w => {
      const v = byWord.get(w);
      if (v === undefined) return `<span class="chip oov" title="Not one of the model's 5,000 words">${esc(w)}</span>`;
      const a = Math.round(12 + Math.abs(v) / mx * 55);
      return `<span class="chip" style="background:color-mix(in srgb,${v >= 0 ? 'var(--pos)' : 'var(--neg)'} ${a}%,var(--surface))">${esc(w)}<small>${v >= 0 ? '+' : ''}${v.toFixed(2)}</small></span>`;
    }).join('');
    const isPos = r.p >= 0.5;
    out.innerHTML = `<div class="verdict"><b style="color:${isPos ? 'var(--pos)' : 'var(--neg)'}">${isPos ? 'Positive' : 'Negative'}</b><span class="muted">${pct(isPos ? r.p : 1 - r.p)} sure</span></div>
      <div class="pbar" role="img" aria-label="${pct(r.p)} positive"><i style="width:${r.p * 100}%"></i></div>
      <div><p class="lbl" style="margin-bottom:6px">After cleaning, stop words and lemmatizing</p><div class="proc">${esc(r.processed) || '<i>nothing left</i>'}</div></div>
      <div><p class="lbl" style="margin-bottom:6px">What each word added (dashed = not in the vocabulary)</p><div class="chips">${chips || '<span class="muted">No words left</span>'}</div></div>`;
  }

  function drawHarness() {
    const W = 560, rowH = 30, m = { l: 150, r: 46, t: 8, b: 26 }, H = m.t + m.b + HARNESS.length * rowH, iw = W - m.l - m.r;
    const x = v => m.l + (v - 0.7) / 0.3 * iw;
    let s = `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy and F1 for eight models">`;
    for (let v = 0.7; v <= 1.0001; v += 0.1) s += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H - m.b}"/><text class="tick" x="${x(v)}" y="${H - 8}" text-anchor="middle">${v.toFixed(1)}</text>`;
    HARNESS.forEach(([name, acc, f1], k) => {
      const y = m.t + k * rowH;
      s += `<text class="lbltxt" x="${m.l - 10}" y="${y + 19}" text-anchor="end" ${k === 0 ? 'font-weight="600"' : ''}>${esc(name)}</text>`;
      s += `<rect x="${m.l}" y="${y + 6}" width="${(x(acc) - m.l).toFixed(1)}" height="8" rx="2" style="fill:var(--gold)"/>`;
      s += `<rect x="${m.l}" y="${y + 16}" width="${(x(f1) - m.l).toFixed(1)}" height="8" rx="2" style="fill:var(--ink)"/>`;
      s += `<text class="tick" x="${x(Math.max(acc, f1)) + 6}" y="${y + 19}">${acc.toFixed(3)}</text>`;
    });
    $('#harness').innerHTML = s + '</svg><p class="muted small">Axis starts at 0.70 so the differences are visible.</p>';
  }

  /* ---------- input ---------- */
  function parseCSV(text) {
    const rows = []; let row = [], f = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) { if (ch === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += ch; continue; }
      if (ch === '"') q = true; else if (ch === ',') { row.push(f); f = ''; }
      else if (ch === '\n' || ch === '\r') { if (ch === '\r' && text[i + 1] === '\n') i++; row.push(f); f = ''; if (row.some(c => c.trim())) rows.push(row); row = []; }
      else f += ch;
    }
    row.push(f); if (row.some(c => c.trim())) rows.push(row);
    return rows;
  }
  async function readFile(file) {
    const msg = $('#fileMsg');
    try {
      const text = await file.text();
      if (/\.csv$/i.test(file.name)) {
        const rows = parseCSV(text);
        if (rows.length < 2) { msg.textContent = 'That CSV has no data rows.'; return; }
        fileRows = rows;
        const head = rows[0], lens = head.map((_, j) => rows.slice(1, 200).reduce((a, r) => a + (r[j] || '').length, 0));
        const best = lens.indexOf(Math.max(...lens));
        $('#colSel').innerHTML = head.map((h, j) => `<option value="${j}"${j === best ? ' selected' : ''}>${esc(h || 'column ' + (j + 1))}</option>`).join('');
        $('#colRow').hidden = false;
        pickColumn();
        msg.textContent = `${file.name}: ${int(rows.length - 1)} rows. Pick the column, then click Read the crowd.`;
      } else {
        fileRows = null; $('#colRow').hidden = true;
        fileComments = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
        msg.textContent = `${file.name}: ${int(fileComments.length)} comments. Click Read the crowd.`;
      }
    } catch (e) { msg.textContent = `Couldn't read ${file.name}: ${e.message}`; }
  }
  function pickColumn() { const j = +$('#colSel').value; fileComments = fileRows.slice(1).map(r => (r[j] || '').trim()).filter(Boolean); }

  function currentComments() {
    if (source === 'sample') return sample.comments;
    if (source === 'paste') return $('#pasteBox').value.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    return fileComments || [];
  }
  function run() {
    const comments = currentComments().slice(0, 50000), msg = $('#runMsg');
    if (comments.length < 1) { msg.textContent = source === 'file' ? 'Choose a file first.' : 'Add at least one comment.'; return; }
    const t0 = performance.now();
    const res = analyze(comments);
    renderBoard(res);
    msg.textContent = `Scored ${int(comments.length)} comments in ${Math.max(1, Math.round(performance.now() - t0))} ms.`;
  }

  function wire() {
    $$('.tabs button').forEach(b => b.addEventListener('click', () => {
      source = b.dataset.src;
      $$('.tabs button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
      ['sample', 'paste', 'file'].forEach(s => $('#src-' + s).hidden = s !== source);
    }));
    $('#runBtn').addEventListener('click', run);
    $('#fileIn').addEventListener('change', e => { const f = e.target.files[0]; if (f) readFile(f); });
    const drop = $('#drop');
    ['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) readFile(f); });
    $('#colSel').addEventListener('change', pickColumn);
    let t; $('#oneBox').addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => { $$('.clist button').forEach(b => b.setAttribute('aria-pressed', 'false')); renderOne($('#oneBox').value); }, 120); });
    document.addEventListener('click', e => { const b = e.target.closest('.clist button'); if (b && current) { const c = current.scored[+b.dataset.i]; explain(c.text, c.i); } });
  }

  async function start() {
    drawHarness();
    wire();
    try {
      const [m, s] = await Promise.all([fetch('data/model.json').then(r => r.json()), fetch('data/sample_comments.json').then(r => r.json())]);
      model = ReelModel.makeModel(m);
      sample = s;
      $('#heroFilm').textContent = s.film;
      $('#aucProba').textContent = m.metrics.roc_auc_proba.toFixed(3);
      $('#pasteBox').value = s.comments.slice(0, 8).join('\n');
      const res = analyze(s.comments);
      renderHero(res);
      renderBoard(res);
      $('#runMsg').textContent = `Showing the sample: ${s.comments.length} comments.`;
    } catch (e) {
      $('#runMsg').textContent = "The model file didn't load. If you opened index.html straight from your disk, serve the folder instead (for example: python -m http.server) or use the Netlify site.";
      $('#runBtn').disabled = true;
      console.error(e);
    }
  }
  start();
})();
