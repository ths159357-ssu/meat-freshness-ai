// 4조 육류 신선도 판단 - 웹 추론 엔진
// colab/1_train_export.py 의 전처리·모델과 같은 계산을 브라우저에서 수행한다.
(function (root) {
  "use strict";

  function median(a) {
    const s = a.slice().sort((x, y) => x - y), n = s.length, h = n >> 1;
    return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
  }
  function softmax(z) {
    const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), t = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / t);
  }

  // 1) 전처리: 로그 → 보관 시작 10분 평균과의 차이 → 5분 이동 중앙값
  function preprocess(M, rows) {
    const C = M.sensors.length, n = rows.length;
    const R = rows.map(r => M.sensors.map(s => Math.log(Math.max(r[s], 1e-6))));
    const nb = Math.min(M.base_minutes, n);
    const base = M.sensors.map((_, c) => R.slice(0, nb).reduce((a, r) => a + r[c], 0) / nb);
    const d = R.map(r => r.map((v, c) => v - base[c]));
    const x = d.map((_, i) => M.sensors.map((__, c) =>
      Math.fround(median(d.slice(Math.max(0, i - 4), i + 1).map(r => r[c])))));   // 학습 코드와 같은 float32
    return x;
  }

  // 2) 온도 누적 지표 tt (Ratkowsky 식, 1분 단위 누적, 빠진 분은 온도를 직선으로 이어 계산)
  function growthTT(M, rows) {
    const rate = T => Math.pow(M.growth_b * Math.max(T - M.growth_tmin, 0), 2) / 60;
    const tt = new Array(rows.length);
    let acc = rate(rows[0].Temperature);
    tt[0] = acc;
    for (let i = 1; i < rows.length; i++) {
      const g = rows[i].Minute - rows[i - 1].Minute, T0 = rows[i - 1].Temperature, T1 = rows[i].Temperature;
      for (let k = 1; k <= g; k++) acc += rate(T0 + (T1 - T0) * k / g);
      tt[i] = acc;
    }
    return tt.map(v => v / Math.log(10));   // 학습 코드처럼 누적 후 ln10으로 나눔
  }

  // 3) 1D-CNN (Conv1D 16,5 → MaxPool 2 → Conv1D 16,3 → GAP → Dense 3 softmax)
  function conv1d(inp, W, b) {        // inp [T][Cin], W [k][Cin][Cout], padding "same", ReLU
    const T = inp.length, k = W.length, Cin = W[0].length, Cout = b.length, pl = (k - 1) >> 1;
    const out = [];
    for (let t = 0; t < T; t++) {
      const o = b.slice();
      for (let j = 0; j < k; j++) {
        const ti = t + j - pl; if (ti < 0 || ti >= T) continue;
        const xi = inp[ti], Wj = W[j];
        for (let c = 0; c < Cin; c++) { const v = xi[c], Wc = Wj[c]; for (let f = 0; f < Cout; f++) o[f] += v * Wc[f]; }
      }
      out.push(o.map(v => (v > 0 ? v : 0)));
    }
    return out;
  }
  function cnnProba(M, win) {
    const z = win.map(r => r.map((v, c) => (v - M.mu[c]) / M.sd[c]));
    const h1 = conv1d(z, M.conv1.W, M.conv1.b);
    const p1 = []; for (let t = 0; t + 1 < h1.length; t += 2) p1.push(h1[t].map((v, f) => Math.max(v, h1[t + 1][f])));
    const h2 = conv1d(p1, M.conv2.W, M.conv2.b);
    const g = h2[0].map((_, f) => h2.reduce((a, r) => a + r[f], 0) / h2.length);
    return softmax(M.dense.b.map((bk, k) => g.reduce((a, v, f) => a + v * M.dense.W[f][k], bk)));
  }

  // 4) 온도 누적 지표 로지스틱 회귀
  function lrProba(M, tt) { return softmax(M.lr.coef.map((c, k) => c * tt + M.lr.intercept[k])); }

  // 5) 센서 수준(30분 평균) + tt 그래디언트 부스팅
  function treeValue(t, x) {
    let i = 0;
    while (!t.leaf[i]) {
      const v = x[t.f[i]];
      i = (Number.isNaN(v) ? t.ml[i] : v <= t.t[i]) ? t.l[i] : t.r[i];
    }
    return t.v[i];
  }
  function hgbProba(M, feat) {
    const raw = M.hgb.baseline.slice();
    for (const it of M.hgb.trees) for (let k = 0; k < it.length; k++) raw[k] += treeValue(it[k], feat);
    return softmax(raw);
  }

  // 30분 이동 평균 (pandas rolling().mean()과 같은 보정 합산 방식 → 학습 코드와 소수점까지 일치)
  function rollingMean(x, w) {
    const n = x.length, C = x[0].length, out = x.map(() => new Array(C));
    for (let c = 0; c < C; c++) {
      let sum = 0, nobs = 0, neg = 0, ca = 0, cr = 0, prev = x[0][c], same = 0;
      const add = v => { nobs++; const y = v - ca, t = sum + y; ca = t - sum - y; sum = t;
        if (v < 0 || Object.is(v, -0)) neg++; if (v === prev) same++; else same = 1; prev = v; };
      const rem = v => { nobs--; const y = -v - cr, t = sum + y; cr = t - sum - y; sum = t;
        if (v < 0 || Object.is(v, -0)) neg--; };
      for (let i = 0; i < n; i++) {
        if (i - w >= 0) rem(x[i - w][c]); add(x[i][c]);   // pandas와 같은 순서: 빼고 더하기
        let r = sum / nobs;
        if (same >= nobs) r = prev; else if (neg === 0 && r < 0) r = 0; else if (neg === nobs && r > 0) r = 0;
        out[i][c] = r;
      }
    }
    return out;
  }

  // 6) 단조 규칙: 더 높은 등급이 10분 연속일 때만 올리고, 내려가지 않음
  function monotone(pred, k) {
    let state = 0, run = 0, cand = null; const out = [];
    for (const p of pred) {
      if (p > state) { run = cand === p ? run + 1 : 1; cand = p; if (run >= k) { state = p; run = 0; } }
      else { run = 0; cand = null; }
      out.push(state);
    }
    return out;
  }

  // 전체 판단: rows = [{Minute, MQ135, MQ3, MQ4, MQ8, Temperature}, ...]
  function judge(M, rowsIn, opts) {
    opts = opts || {};
    const rows = rowsIn.slice().sort((a, b) => a.Minute - b.Minute);
    const x = preprocess(M, rows), tt = growthTT(M, rows), n = rows.length, C = M.sensors.length;
    const offs = []; for (let o = M.window - M.step; o >= 0; o -= M.step) offs.push(o);
    const level = rollingMean(x, 30);
    const from = opts.lastOnly ? n - 1 : 0, probs = [], parts = [];
    for (let i = from; i < n; i++) {
      const win = offs.map(o => x[Math.max(i - o, 0)]);
      const lv = level[i];
      const pc = cnnProba(M, win), pl = lrProba(M, tt[i]), ph = hgbProba(M, lv.concat([tt[i]]));
      probs.push(pc.map((v, k) => (v + pl[k] + ph[k]) / 3)); parts.push({ cnn: pc, lr: pl, hgb: ph });
    }
    const argmax = p => p.indexOf(Math.max(...p));
    const grades = opts.lastOnly ? [argmax(probs[0])] : monotone(probs.map(argmax), M.monotone_minutes);
    const last = probs.length - 1;
    return { grade: grades[last], className: M.classes[grades[last]], prob: probs[last], parts: parts[last],
             grades, probs, minutes: rows.slice(from).map(r => r.Minute), tt: tt[n - 1], n };
  }

  // 엑셀/CSV 표 → rows (열 이름 대소문자·공백 무시, "분"/"온도" 한글 열 이름도 허용)
  function rowsFromTable(M, table) {
    const alias = { minute: "Minute", "분": "Minute", "경과분": "Minute", "경과 분": "Minute",
                    temperature: "Temperature", "온도": "Temperature", "온도(°c)": "Temperature", "온도(℃)": "Temperature" };
    M.sensors.forEach(s => (alias[s.toLowerCase()] = s));
    const need = ["Minute", "Temperature"].concat(M.sensors), rows = [];
    for (const r of table) {
      const o = {};
      for (const k in r) { const key = alias[String(k).trim().toLowerCase()]; if (key) o[key] = Number(r[k]); }
      if (need.every(k => Number.isFinite(o[k]))) rows.push(o);
    }
    const missing = table.length ? need.filter(k => !Object.keys(table[0]).some(c => alias[String(c).trim().toLowerCase()] === k)) : need;
    return { rows, missing };
  }

  const api = { judge, rowsFromTable, preprocess, growthTT, cnnProba, lrProba, hgbProba, monotone };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.SpoilageEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
