function buildWorkbook(spec){
  /* port of generate-mto build_mto.py, function by function: same cells, formulas and styles */
  const wb = new ExcelJS.Workbook(); wb.calcProperties = {fullCalcOnLoad: true};
  const FONT = 'Times New Roman', BLUE = 'FF0000FF', GREEN = 'FF008000';
  const thin = {style: 'thin'}, dbl = {style: 'double'};
  const HDR_B = {left: thin, right: thin, top: thin, bottom: dbl}, SUB_B = {top: thin, bottom: thin}, TOT_B = {top: thin, bottom: dbl};
  const fill = argb => ({type: 'pattern', pattern: 'solid', fgColor: {argb}});
  const RED_FILL = fill('FFFFC7CE'), PURPLE_FILL = fill('FFE4D7F5'), HDR_FILL = fill('FFEDEDED');
  const f = (sz = 11, b = false, color = null, u = null) => { const o = {name: FONT, size: sz, bold: b}; if (color) o.color = {argb: color}; if (u) o.underline = u; return o; };
  const fx = s => ({formula: s.replace(/^=/, '')});
  const q = s => `'${s.replace(/'/g, "''")}'`;
  const setPage = (ws, titles) => { ws.pageSetup = Object.assign(ws.pageSetup || {}, {orientation: 'landscape', fitToPage: true, fitToWidth: 1, fitToHeight: 0}); if (titles) ws.pageSetup.printTitlesRow = titles; };
  const titleBlock = (ws, h, vcol) => [["CLIENT'S : ", 'client'], ['PROJECT : ', 'project'], ['DESCRIPTION : ', 'description'], ['CONTRACTOR : ', 'contractor'], ['DATE : ', 'date']].forEach(([lab, key], i) => {
    const a = ws.getCell(i + 1, 1); a.value = lab; a.font = f(11, true); const c = ws.getCell(i + 1, vcol); c.value = h[key] || ''; c.font = f(); });
  const UOMS = ['m2', 'm', 'pcs'];
  const kind = r => r.kind || (r.wid && r.thk && !r.kgm ? 'plate' : (r.len ? 'length' : 'piece'));
  const UNIT_COLS = ['Item', 'Material Section', null, null, null, null, 'Category', 'Grade', 'Description', 'Mark', 'Ref. Mark', 'Unit wt. (kg/m, kg/m2) / kg/no', 'Surface area (m2/m)',
    'Length (mm)', 'Width (mm)', 'Thk (mm)', 'Qty (no)', 'Set (no)', 'Weight (kg)', 'Painted area (m2)', 'Run (m)', 'Area (m2)', 'kg/m run', 'Drawings', 'Cautious'];
  const UNIT_W = [6, 7, 3, 7, 3, 7, 9, 9, 58, 8, 9, 13, 12, 11, 10, 8, 8, 7, 13, 13, 11, 11, 10, 70, 60];
  const NC = UNIT_COLS.length, SRC = NC - 1, CAUT = NC;
  const WT = 19, PAINT = 20, RUN = 21, AREA = 22, KGM = 23;
  const FIELD_COL = {cat: 7, grade: 8, desc: 9, mark: 10, kgm: 12, m2m: 13, len: 14, wid: 15, thk: 16, qty: 17, set: 18};
  const INPUT_COLS = [12, 13, 14, 15, 16, 17, 18];
  const NUMFMT = {12: '0.0##', 13: '0.00#', 14: '#,##0', 15: '#,##0', 16: '0.#', 17: '#,##0', 18: '#,##0', 19: '#,##0.0', 20: '#,##0.0', 21: '#,##0.00', 22: '#,##0.00', 23: '0.0'};
  const CENTER = [1, 7, 8, 10, 11, 17, 18, 20, 21, 22, 23];

  /* only what the drawings show (build_mto.py prune / uses): drop categories with no row in any unit, hide unused quantity columns */
  { const used = spec.categories.filter(c => spec.units.some(u => ((u.rows || {})[c.code] || []).length));
    if (used.length && used.length !== spec.categories.length){ const keep = new Set(used.map(c => c.code));
      spec = Object.assign({}, spec, {categories: used, units: spec.units.map(u => Object.assign({}, u, {rows: Object.fromEntries(Object.entries(u.rows || {}).filter(([k]) => keep.has(k)))}))}); } }
  const allRows = spec.units.flatMap(u => Object.values(u.rows || {}).flat());
  const hasRun = allRows.some(r => r.run || (kind(r) === 'measure' && r.uom === 'm'));
  const hasArea = allRows.some(r => kind(r) === 'area' || (kind(r) === 'measure' && r.uom === 'm2'));
  const codes = spec.categories.map(c => c.code);
  if (new Set(codes).size !== codes.length) throw new Error('duplicate category codes');
  for (const u of spec.units){ const bad = Object.keys(u.rows || {}).filter(k => !codes.includes(k)); if (bad.length) throw new Error(`${u.name}: rows under unknown categories ${bad}`); }

  /* Excel sheet names: max 31 chars, none of []:*?/\ (a longer name is cut and every Summary formula to it turns #REF!) - same as build_mto.py */
  { const used = new Set(); spec = Object.assign({}, spec, {units: spec.units.map(u => {
      let nm = u.name.split('').filter(ch => !'[]:*?/\\'.includes(ch)).join('').trim().slice(0, 31).trim(); const base = nm; let k = 2;
      while (used.has(nm.toLowerCase()) || nm.toLowerCase() === 'summary'){ nm = `${base.slice(0, 31 - String(k).length - 1)}~${k}`; k++; }
      used.add(nm.toLowerCase()); return nm === u.name ? u : Object.assign({}, u, {name: nm}); })}); }
  const putRow = (ws, r, n, row, where) => {
    const k = kind(row);
    if (k === 'measure' && !UOMS.includes(row.uom)) throw new Error(`${where}: measure row needs uom in ${UOMS}`);
    const amt = k === 'measure' && row.amt !== undefined && row.amt !== null ? row.amt : null;   // measure row summed per level
    const vals = {1: n, 7: row.cat, 8: row.grade, 9: row.desc, 10: row.mark, 11: row.ref, 12: row.kgm, 13: row.m2m, 14: row.len, 15: row.wid, 16: row.thk,
      17: amt !== null && row.uom === 'pcs' ? amt : (row.qty ?? 1), 18: row.set ?? 1, [SRC]: row.src, [CAUT]: row.caut};
    if (k === 'length' && row.d !== undefined && row.d !== null){ vals[2] = String(row.d); vals[3] = 'x'; vals[4] = String(row.w ?? '');
      if (row.t !== undefined && row.t !== null && row.t !== ''){ vals[5] = 'x'; vals[6] = String(row.t); } }
    for (const cs in vals){ const c = +cs, v = vals[cs]; if (v === undefined || v === null) continue;
      const cell = ws.getCell(r, c); cell.value = v; cell.font = f(c === SRC || c === CAUT ? 9 : 11, false, INPUT_COLS.includes(c) ? BLUE : null);
      if (NUMFMT[c]) cell.numFmt = NUMFMT[c];
      if (CENTER.includes(c)) cell.alignment = {horizontal: 'center', vertical: 'top'};
      else if (c === SRC || c === CAUT) cell.alignment = {wrapText: false, vertical: 'top'}; }
    for (const [fk, fl] of [['scaled', PURPLE_FILL], ['assumed', RED_FILL]]) for (const fld of row[fk] || []) ws.getCell(r, FIELD_COL[fld]).fill = fl;
    let fm;
    if (k === 'length') fm = `=L${r}*N${r}/1000*Q${r}*R${r}`;
    else if (k === 'plate') fm = `=N${r}/1000*O${r}/1000*P${r}*${row.dens ?? 7.85}*Q${r}*R${r}`;
    else if (k === 'area') fm = `=L${r}*N${r}/1000*O${r}/1000*Q${r}*R${r}`;
    else if (k === 'measure') fm = null;
    else fm = `=L${r}*Q${r}*R${r}`;
    if (fm){ const p = ws.getCell(r, WT); p.value = fx(fm); p.font = f(); p.numFmt = NUMFMT[WT]; p.alignment = {vertical: 'top'}; }
    let run = row.run || (k === 'measure' && row.uom === 'm') ? fx(`=N${r}/1000*Q${r}*R${r}`) : null;
    let area = k === 'area' || (k === 'measure' && row.uom === 'm2') ? fx(`=N${r}/1000*O${r}/1000*Q${r}*R${r}`) : null;
    if (amt !== null && row.uom !== 'pcs'){ if (row.uom === 'm'){ run = amt; area = null; } else { run = null; area = amt; } }
    for (const [c, v] of [[RUN, run], [AREA, area]]) if (v !== null){ const cell = ws.getCell(r, c); cell.value = v; cell.font = f(11, false, amt !== null ? BLUE : null); cell.numFmt = NUMFMT[c]; cell.alignment = {horizontal: 'center', vertical: 'top'}; }
    let pa = null;
    if (k === 'length' && row.m2m !== undefined && row.m2m !== null) pa = `=M${r}*N${r}/1000*Q${r}*R${r}`;
    else if (k === 'plate') pa = `=2*N${r}/1000*O${r}/1000*Q${r}*R${r}`;
    if (pa){ const c = ws.getCell(r, PAINT); c.value = fx(pa); c.font = f(); c.numFmt = NUMFMT[PAINT]; c.alignment = {horizontal: 'center', vertical: 'top'}; }
    for (let c = 1; c <= NC; c++){ const cell = ws.getCell(r, c); const al = cell.alignment || {};
      cell.alignment = {horizontal: al.horizontal, vertical: 'top', wrapText: al.wrapText}; }
  };
  const putSub = (ws, r, label, rng, extra = [], total = false) => {   // S..V = SUM(rng) + extra rows; W = kg/m run
    ws.getCell(r, 9).value = label;
    for (const c of [WT, PAINT, RUN, AREA]){ const L = colL(c);
      const parts = (rng && rng[1] >= rng[0] ? [`SUM(${L}${rng[0]}:${L}${rng[1]})`] : []).concat(extra.map(x => `${L}${x}`));
      ws.getCell(r, c).value = parts.length ? fx('=' + parts.join('+')) : 0; }
    if (!total) ws.getCell(r, KGM).value = fx(`=IF(AND(U${r}>0,S${r}>0),S${r}/U${r},"")`);
    for (let c = 1; c <= NC; c++){ const cell = ws.getCell(r, c); cell.border = total ? TOT_B : SUB_B; cell.font = total ? f(12, true) : f(11, true); }
    ws.getCell(r, WT).numFmt = ws.getCell(r, PAINT).numFmt = '#,##0';
    ws.getCell(r, RUN).numFmt = ws.getCell(r, AREA).numFmt = '#,##0.0';
    if (!total) ws.getCell(r, KGM).numFmt = NUMFMT[KGM];
  };
  const summ = wb.addWorksheet('Summary ');
  const links = {};
  for (const unit of spec.units){
    const ws = wb.addWorksheet(unit.name);
    const note = unit.note ? `   (${unit.note})` : '';
    const h6 = ws.getCell(1, 1); h6.value = `QTY-BREAKDOWN : ${unit.name}${note}`; h6.font = f(12, true);
    const key = ws.getCell(1, CAUT); key.value = 'Red fill = ASSUMED   Purple fill = SCALED   Blue font = input'; key.font = f(9, true); key.alignment = {horizontal: 'left'};
    UNIT_COLS.forEach((t, i) => { const cell = ws.getCell(2, i + 1); if (t !== null) cell.value = t; cell.font = f(11, true); cell.border = HDR_B; cell.fill = HDR_FILL;
      cell.alignment = {horizontal: 'center', vertical: 'middle', wrapText: true}; });
    ws.mergeCells('B2:F2'); ws.getRow(2).height = 32;
    setPage(ws, '2:2');
    UNIT_W.forEach((w, i) => ws.getColumn(i + 1).width = w === 9 ? 9.0001 : w);  // ExcelJS omits width 9 (its default)
    ws.getColumn(SRC).hidden = true;                                              // Drawings: hidden by default
    ws.views = [{state: 'frozen', xSplit: 0, ySplit: 2, topLeftCell: 'A3'}];
    if (!hasRun){ ws.getColumn(RUN).hidden = true; ws.getColumn(KGM).hidden = true; }
    if (!hasArea) ws.getColumn(AREA).hidden = true;
    const rowsOf = unit.rows || {};
    let r = 3; const subs = {};
    if (!Object.values(rowsOf).some(rs => rs.some(row => row.sec))){
      for (const cat of spec.categories){
        const code = cat.code;
        const a = ws.getCell(r, 1); a.value = code; a.font = f(13, true); a.alignment = {horizontal: 'center'};
        const t = ws.getCell(r, 9); t.value = cat.title; t.font = f(13, true, null, 'single');
        const dwg = (unit.dwg || {})[code] || cat.dwg || '';
        const dl = ws.getCell(r + 1, 9); dl.value = dwg ? `Dwg No: ${dwg}` : 'Dwg No: -'; dl.font = f(11, true);
        r += 2; const first = r;
        (rowsOf[code] || []).forEach((row, i) => { putRow(ws, r, i + 1, row, `${unit.name} ${code}`); r++; });
        putSub(ws, r, `Sub-Total ${code}`, [first, r - 1]);
        subs[code] = r; r += 2;
      }
    } else {
      const secs = [...(unit.secs || [])];
      for (const rs of Object.values(rowsOf)) for (const row of rs){ const s = row.sec || '-'; if (!secs.includes(s)) secs.push(s); }
      const catSubs = Object.fromEntries(spec.categories.map(c => [c.code, []]));
      for (const sec of secs){
        const st = ws.getCell(r, 9); st.value = sec.toUpperCase(); st.font = f(13, true, null, 'single'); r++;
        const secSubs = [];
        for (const cat of spec.categories){
          const code = cat.code;
          const rs = (rowsOf[code] || []).filter(row => (row.sec || '-') === sec);
          if (!rs.length) continue;
          const a = ws.getCell(r, 1); a.value = code; a.font = f(12, true); a.alignment = {horizontal: 'center'};
          const t = ws.getCell(r, 9); t.value = cat.title; t.font = f(11, true, null, 'single'); r++;
          const first = r; let n = 0;
          for (const row of rs.filter(x => !x.sub)){ putRow(ws, r, ++n, row, `${unit.name} ${code}`); r++; }
          const direct = [first, r - 1];
          const groups = []; for (const row of rs) if (row.sub && !groups.includes(row.sub)) groups.push(row.sub);
          const extra = [];
          for (const sb of groups){
            const gt = ws.getCell(r, 9); gt.value = sb; gt.font = f(11, true, null, 'single'); r++;
            const g0 = r;
            rs.filter(x => x.sub === sb).forEach((row, i) => { putRow(ws, r, i + 1, row, `${unit.name} ${code} ${sb}`); r++; });
            putSub(ws, r, `Sub-Total ${sb}`, [g0, r - 1]); extra.push(r); r++;
          }
          putSub(ws, r, `Sub-Total ${code} - ${sec}`, direct[1] >= direct[0] ? direct : null, extra);
          secSubs.push(r); catSubs[code].push(r); r += 2;
        }
        putSub(ws, r, `TOTAL ${sec} (kg)`, null, secSubs, true); r += 3;
      }
      const ct = ws.getCell(r, 9); ct.value = 'CATEGORY TOTALS (all storeys)'; ct.font = f(13, true, null, 'single'); r++;
      for (const cat of spec.categories){
        const code = cat.code;
        const a = ws.getCell(r, 1); a.value = code; a.font = f(12, true); a.alignment = {horizontal: 'center'};
        putSub(ws, r, `Sub-Total ${code}  ${cat.title}`, null, catSubs[code]);
        const dwg = (unit.dwg || {})[code] || cat.dwg || '';
        if (dwg){ const d = ws.getCell(r, SRC); d.value = `Dwg No: ${dwg}`; d.font = f(9); }
        subs[code] = r; r++;
      }
      r++;
    }
    const total = r;
    putSub(ws, r, `TOTAL ${unit.name} (kg)`, null, Object.values(subs), true);
    ws.pageSetup.printArea = `A1:W${total}`;
    links[unit.name] = [subs, total];
  }

  // summary
  const ws = summ;
  const cur = spec.currency || 'RM', units = spec.units.map(u => u.name), n = units.length;
  const tcol = 3 + n, ucol = 4 + n, rcol = 5 + n, acol = 6 + n, T = colL(tcol), R = colL(rcol), A = colL(acol), last = colL(acol);
  titleBlock(ws, spec.header, 2);
  ['Sl. No.', 'Description', ...units, 'Total (kg)', 'Unit', `Rate (${cur})`, `Amount (${cur})`, 'Painted area (m2)', 'Run (m)', 'Area (m2)', 'kg/m run'].forEach((t, i) => {
    const cell = ws.getCell(7, i + 1); cell.value = t; cell.font = f(11, true); cell.border = HDR_B; cell.fill = HDR_FILL;
    cell.alignment = {horizontal: 'center', vertical: 'middle', wrapText: true}; });
  ws.getRow(7).height = 32;
  setPage(ws, null);
  [16, 80, ...Array(n).fill(13), 14, 7, 11, 13, 16, 11, 11, 10].forEach((w, i) => ws.getColumn(i + 1).width = w === 9 ? 9.0001 : w);  // ExcelJS omits width 9 (its default)
  if (!hasRun){ ws.getColumn(acol + 2).hidden = true; ws.getColumn(acol + 4).hidden = true; }
  if (!hasArea) ws.getColumn(acol + 3).hidden = true;
  const rates = spec.rates || {};
  let r = 8;
  for (const cat of spec.categories){
    const code = cat.code;
    const a = ws.getCell(r, 1); a.value = code; a.font = f(13, true); a.alignment = {horizontal: 'center'};
    const b = ws.getCell(r, 2); b.value = cat.title; b.font = f(11, true); b.alignment = {wrapText: true, vertical: 'middle'};
    units.forEach((u, j) => { const c = ws.getCell(r, 3 + j); c.value = fx(`=${q(u)}!S${links[u][0][code]}`); c.font = f(11, false, GREEN); c.numFmt = '#,##0'; });
    const c = ws.getCell(r, tcol); c.value = fx(`=SUM(C${r}:${colL(2 + n)}${r})`); c.font = f(11, true); c.numFmt = '#,##0';
    ws.getCell(r, ucol).value = 'kg'; ws.getCell(r, ucol).font = f();
    const pc = ws.getCell(r, acol + 1); pc.value = fx('=' + units.map(u => `${q(u)}!T${links[u][0][code]}`).join('+')); pc.font = f(11, false, GREEN); pc.numFmt = '#,##0.0';
    for (const [k, col] of [[2, 'U'], [3, 'V']]){ const cc = ws.getCell(r, acol + k); cc.value = fx('=' + units.map(u => `${q(u)}!${col}${links[u][0][code]}`).join('+')); cc.font = f(11, false, GREEN); cc.numFmt = '#,##0.0'; }
    const rn = colL(acol + 2);
    const km = ws.getCell(r, acol + 4); km.value = fx(`=IF(AND(${rn}${r}>0,${T}${r}>0),${T}${r}/${rn}${r},"")`); km.font = f(11, true); km.numFmt = '0.0';
    if (code in rates){ const rc = ws.getCell(r, rcol); rc.value = rates[code]; rc.font = f(11, false, BLUE);
      const am = ws.getCell(r, acol); am.value = fx(`=${T}${r}*${R}${r}`); am.font = f(); am.numFmt = '#,##0.00'; }
    r++;
  }
  const first = 8, lastcat = r - 1;
  ws.getCell(r, 2).value = spec.unit_label === undefined ? 'Sub-Total per unit (kg)' : `Sub-Total per ${spec.unit_label} (kg)`;
  for (let c = 3; c <= tcol; c++){ const cl = colL(c); const cell = ws.getCell(r, c); cell.value = fx(`=SUM(${cl}${first}:${cl}${lastcat})`); cell.numFmt = '#,##0'; }
  if (Object.keys(rates).length){ const cell = ws.getCell(r, acol); cell.value = fx(`=SUM(${A}${first}:${A}${lastcat})`); cell.numFmt = '#,##0.00'; }
  for (let c = 1; c <= acol + 4; c++){ const cell = ws.getCell(r, c); cell.border = SUB_B; cell.font = f(11, true); }
  const sub = r;
  for (const k of [1, 2, 3]){ const cl = colL(acol + k); const ps = ws.getCell(r, acol + k); ps.value = fx(`=SUM(${cl}${first}:${cl}${lastcat})`); ps.font = f(11, true); ps.numFmt = '#,##0.0'; }
  r++;
  const word = spec.unit_label || 'unit';
  const ck = ws.getCell(r, 2); ck.value = `Check: ${word} sheet total (kg)`; ck.font = f(10);
  units.forEach((u, j) => { const c = ws.getCell(r, 3 + j); c.value = fx(`=${q(u)}!S${links[u][1]}`); c.font = f(10, false, GREEN); c.numFmt = '#,##0'; });
  r++; const pct = r;
  const al = ws.getCell(r, 2); al.value = 'Connection / misc. allowance (%) - 0 when plates and bolts are itemised'; al.font = f();
  const pcc = ws.getCell(r, tcol); const aPct = spec.allowance || 0; pcc.value = aPct >= 1 ? aPct / 100 : aPct; pcc.font = f(11, false, BLUE); pcc.numFmt = '0.0%';  // spec gives % (5 = 5 %); a fraction is accepted too
  r++;
  const al2 = ws.getCell(r, 2); al2.value = 'Allowance (kg)'; al2.font = f();
  for (let c = 3; c <= tcol; c++){ const cl = colL(c); const cell = ws.getCell(r, c); cell.value = fx(`=${cl}${sub}*$${T}$${pct}`); cell.font = f(); cell.numFmt = '#,##0'; }
  const allow = r; r += 2;
  const label = spec.grand_total_label || `${n} UNITS`;
  ws.getCell(r, 2).value = `GRAND TOTAL - ${label} (kg)`;
  const g1 = ws.getCell(r, tcol); g1.value = fx(`=${T}${sub}+${T}${allow}`); g1.numFmt = '#,##0';
  ws.getCell(r + 1, 2).value = `GRAND TOTAL - ${label} (tonnes)`;
  const g2 = ws.getCell(r + 1, tcol); g2.value = fx(`=${T}${r}/1000`); g2.numFmt = '#,##0.00';
  for (const rr of [r, r + 1]) for (let c = 1; c <= acol + 4; c++){ const cell = ws.getCell(rr, c); cell.border = TOT_B; cell.font = f(12, true); }
  r += 2;
  const sections = [['LEFT OUT OF THIS TAKE-OFF', (spec.left_out && spec.left_out.length) ? spec.left_out : ['Nothing left out.']]];   // the curator: left out, cautious, decisions, assumptions
  if (spec.cautions && spec.cautions.length) sections.push(['CAUTIOUS', spec.cautions]);
  if (spec.decisions && spec.decisions.length) sections.push(['DECISIONS TAKEN', spec.decisions]);
  sections.push(['ASSUMPTIONS / ITEMS TO VERIFY', spec.assumptions || []]);
  for (const [title, items] of sections){
    r += 2;
    const ah = ws.getCell(r, 1); ah.value = title; ah.font = f(12, true);
    for (const a of items){
      r++;
      const d = ws.getCell(r, 1); d.value = '-'; d.font = f(10); d.alignment = {horizontal: 'right', vertical: 'top'};
      const c = ws.getCell(r, 2); c.value = a; c.font = f(10); c.alignment = {wrapText: true, vertical: 'top'};
      ws.mergeCells(`B${r}:${last}${r}`);
      ws.getRow(r).height = 15 * Math.max(2, Math.ceil(a.length / 170));
    }
  }
  return wb;
}
