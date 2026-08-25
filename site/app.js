(() => {
  const revealElements = [...document.querySelectorAll('.reveal')];
  const showAllReveals = () => revealElements.forEach(el => el.classList.add('visible'));

  try {
    if (typeof IntersectionObserver === 'function') {
      const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: 0.12 });
      revealElements.forEach(el => observer.observe(el));
      window.__klRevealAnimationReady = true;
    } else {
      showAllReveals();
    }
  } catch (_error) {
    // Animation is progressive enhancement: content must stay readable even if
    // observer setup is unavailable or fails in an older browser.
    showAllReveals();
  }

  const svg = document.getElementById('knowledge-field');
  if (!svg) return;

  const NS = 'http://www.w3.org/2000/svg';
  const base = document.getElementById('field-base');
  const reveal = document.getElementById('field-reveal');
  const ring = document.getElementById('lens-ring');
  const core = document.getElementById('lens-core');
  const clipCircle = document.getElementById('lens-clip-circle');
  const demo = svg.closest('.lens-demo');
  const readout = document.getElementById('evidence-readout');
  if (!base || !reveal || !ring || !core || !clipCircle || !demo || !readout) return;

  const nodes = [
    { id:'docs', label:'Documents', x:110, y:270, r:15 },
    { id:'transformer', label:'Transformer', x:285, y:120, r:18 },
    { id:'attention', label:'Attention', x:420, y:235, r:22, master:true },
    { id:'parallel', label:'Parallel Training', x:620, y:145, r:17 },
    { id:'recurrence', label:'Recurrence', x:630, y:340, r:16 },
    { id:'complexity', label:'Complexity', x:385, y:410, r:15 },
    { id:'source', label:'Source Evidence', x:185, y:430, r:17 },
    { id:'rag', label:'Graph RAG', x:545, y:475, r:16 },
  ];

  const edges = [
    { a:'docs', b:'transformer', relation:'contains', evidence:'The uploaded paper introduces the Transformer architecture.', source:'transformer-paper.pdf', location:'p. 1', confidence:'0.97' },
    { a:'transformer', b:'attention', relation:'uses', evidence:'The architecture is based solely on attention mechanisms.', source:'transformer-paper.pdf', location:'p. 1', confidence:'0.98' },
    { a:'attention', b:'parallel', relation:'enables', evidence:'Attention removes recurrent dependencies between positions, improving parallelization.', source:'transformer-paper.pdf', location:'p. 4', confidence:'0.94' },
    { a:'attention', b:'recurrence', relation:'replaces', evidence:'Self-attention is used in place of recurrent layers.', source:'transformer-paper.pdf', location:'p. 2', confidence:'0.93' },
    { a:'attention', b:'complexity', relation:'changes', evidence:'Attention trades sequential operations for matrix operations.', source:'analysis-notes.md', location:'chunk 8', confidence:'0.86' },
    { a:'source', b:'attention', relation:'supports', evidence:'Evidence remains attached to the claim that produced the edge.', source:'KnowledgeLens graph', location:'claim 04', confidence:'1.00' },
    { a:'source', b:'docs', relation:'originates from', evidence:'Each extracted claim remembers its source document and location.', source:'KnowledgeLens graph', location:'claim 01', confidence:'1.00' },
    { a:'attention', b:'rag', relation:'retrieved by', evidence:'Relevant graph claims and paths become grounded context for chat.', source:'retrieval.py', location:'chunk 21', confidence:'0.91' },
    { a:'complexity', b:'rag', relation:'queried through', evidence:'Graph retrieval can surface multiple connected concepts.', source:'retrieval.py', location:'chunk 25', confidence:'0.89' },
    { a:'parallel', b:'rag', relation:'explained by', evidence:'Graph-grounded answers cite source-backed relationships.', source:'KnowledgeLens graph', location:'claim 10', confidence:'0.92' },
    { a:'recurrence', b:'rag', relation:'contrasted in', evidence:'Connected claims allow questions about differences and paths.', source:'KnowledgeLens graph', location:'claim 11', confidence:'0.87' },
  ];

  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const create = (tag, attrs, parent) => {
    const el = document.createElementNS(NS, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
    parent.appendChild(el);
    return el;
  };

  edges.forEach((edge, index) => {
    const a = byId[edge.a], b = byId[edge.b];
    create('line', { x1:a.x, y1:a.y, x2:b.x, y2:b.y, class:'edge', 'data-edge':index }, base);
    create('line', { x1:a.x, y1:a.y, x2:b.x, y2:b.y, class:'edge', 'data-edge':index }, reveal);
    const tx = (a.x + b.x) / 2, ty = (a.y + b.y) / 2;
    const label = create('text', { x:tx, y:ty - 7, class:'relation', 'text-anchor':'middle' }, reveal);
    label.textContent = edge.relation.toUpperCase();
  });

  nodes.forEach(node => {
    const cls = `node${node.master ? ' master' : ''}`;
    create('circle', { cx:node.x, cy:node.y, r:node.r, class:cls }, base);
    const circle = create('circle', { cx:node.x, cy:node.y, r:node.r, class:cls, tabindex:'0', role:'button', 'aria-label':node.label }, reveal);
    const text = create('text', { x:node.x, y:node.y - node.r - 10, 'text-anchor':'middle' }, reveal);
    text.textContent = node.label;
    circle.addEventListener('focus', () => showNodeEvidence(node.id));
  });

  let activeEdge = 2;
  const updateReadout = edgeIndex => {
    const edge = edges[edgeIndex];
    if (!edge) return;
    activeEdge = edgeIndex;
    const a = byId[edge.a], b = byId[edge.b];
    readout.innerHTML = `
      <div class="readout-index">CLAIM ${String(edgeIndex + 1).padStart(2,'0')} / ${String(edges.length).padStart(2,'0')}</div>
      <strong>${a.label} <span>${edge.relation}</span> ${b.label}</strong>
      <p>“${edge.evidence}”</p>
      <footer><span>${edge.source}</span><b>${edge.location}</b><small>${edge.confidence} confidence</small></footer>`;
  };

  const showNodeEvidence = id => {
    const idx = edges.findIndex(edge => edge.a === id || edge.b === id);
    if (idx >= 0) updateReadout(idx);
  };

  const nearestEdge = (x, y) => {
    let best = { idx: activeEdge, distance: Infinity };
    edges.forEach((edge, idx) => {
      const a = byId[edge.a], b = byId[edge.b];
      const vx = b.x - a.x, vy = b.y - a.y;
      const wx = x - a.x, wy = y - a.y;
      const c2 = vx*vx + vy*vy;
      const t = Math.max(0, Math.min(1, c2 ? (wx*vx + wy*vy) / c2 : 0));
      const px = a.x + t*vx, py = a.y + t*vy;
      const d = Math.hypot(x-px, y-py);
      if (d < best.distance) best = { idx, distance:d };
    });
    return best;
  };

  const moveLens = (clientX, clientY) => {
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const local = point.matrixTransform(ctm.inverse());
    const x = local.x;
    const y = local.y;
    [ring, core, clipCircle].forEach(el => { el.setAttribute('cx', x); el.setAttribute('cy', y); });
    const near = nearestEdge(x, y);
    if (near.distance < 95 && near.idx !== activeEdge) updateReadout(near.idx);
  };

  demo.addEventListener('pointermove', event => moveLens(event.clientX, event.clientY));
  demo.addEventListener('pointerdown', event => moveLens(event.clientX, event.clientY));
  updateReadout(activeEdge);

  const steps = [...document.querySelectorAll('.trace-step')];
  const beam = document.getElementById('trace-beam');
  const updateTrace = () => {
    if (!beam || !steps.length) return;
    let closest = 0, min = Infinity;
    steps.forEach((step, idx) => {
      const rect = step.getBoundingClientRect();
      const d = Math.abs(rect.top - window.innerHeight * .35);
      if (d < min) { min = d; closest = idx; }
      step.style.opacity = d < window.innerHeight * .65 ? '1' : '.66';
    });
    beam.style.filter = `hue-rotate(${closest * 5}deg)`;
  };
  window.addEventListener('scroll', updateTrace, { passive:true });
  updateTrace();
})();
