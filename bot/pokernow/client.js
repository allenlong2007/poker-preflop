// PokerNow page script for the match bot (experiment 31), advice-only. Served by bot/pokernow/server.py.
// Reads the game log, asks the local server for a decision and shows a labeled overlay. It never
// clicks: you make every move yourself. The hero is the player in the seat locked on first load.
(() => {
  const SERVER = (document.currentScript && new URL(document.currentScript.src).origin) || "http://127.0.0.1:8765";
  const POLL_MS = 1500;
  if (window.__pnbot) { clearInterval(window.__pnbot.timer); window.__pnbot.root?.remove(); }
  const gameId = (location.pathname.match(/\/games\/([^/?#]+)/) || [])[1];
  const KEY = (k) => `pnbot-${k}-${gameId}`;
  const store = {
    get: (k, d) => { try { const v = localStorage.getItem(KEY(k)); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set: (k, v) => { try { localStorage.setItem(KEY(k), JSON.stringify(v)); } catch {} },
  };
  const state = { seat: store.get("seat", null), busy: false, logUrl: null };

  // ------------------------------------------------------------------ overlay
  const root = document.createElement("div");
  root.id = "pnbot";
  root.innerHTML = `
    <style>
      #pnbot{position:fixed;top:12px;right:12px;width:310px;z-index:2147483647;background:#14171c;color:#e8e8e8;
        font:12px/1.45 -apple-system,system-ui,sans-serif;border:1px solid #2c323b;border-radius:10px;
        box-shadow:0 8px 24px rgba(0,0,0,.45);overflow:hidden}
      #pnbot .hd{display:flex;align-items:center;gap:8px;padding:8px 10px;background:#1c2027;cursor:move}
      #pnbot .hd b{flex:1;font-size:12px;letter-spacing:.02em}
      #pnbot .dot{width:8px;height:8px;border-radius:50%;background:#777}
      #pnbot .sec{padding:7px 10px;border-top:1px solid #262b33}
      #pnbot .t{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#8b95a3;margin-bottom:3px}
      #pnbot .row{display:flex;justify-content:space-between;gap:8px}
      #pnbot .row > span:first-child{color:#9aa4b2}
      #pnbot .v{font-variant-numeric:tabular-nums;text-align:right}
      #pnbot .bb{color:#8b95a3}
      #pnbot .cards{font-size:15px;letter-spacing:.04em}
      #pnbot .red{color:#ff6b6b}
      #pnbot .rec{font-size:15px;font-weight:600;margin:2px 0 4px}
      #pnbot .mix{color:#aab3bf;font-size:11px}
      #pnbot select{font:inherit;background:#262c35;color:#e8e8e8;border:1px solid #39414c;border-radius:6px;padding:2px 6px}
      #pnbot .min{border:none;background:none;padding:0 2px;color:#9aa4b2;font:inherit;cursor:pointer}
      #pnbot.collapsed .body{display:none}
      #pnbot .err{color:#ff8f8f;white-space:pre-wrap}
    </style>
    <div class="hd"><span class="dot"></span><b>Poker bot (advice only)</b><button class="min" data-a="min">–</button></div>
    <div class="body">
      <div class="sec"><div class="t">Table</div>
        <div class="row"><span>Hand</span><span class="v" data-f="hand">–</span></div>
        <div class="row"><span>Street</span><span class="v" data-f="street">–</span></div>
        <div class="row"><span>Button (dealer)</span><span class="v" data-f="button">–</span></div>
      </div>
      <div class="sec"><div class="t" data-f="youT">You</div>
        <div class="row"><span>Position</span><span class="v" data-f="youPos">–</span></div>
        <div class="row"><span>Stack</span><span class="v" data-f="youStack">–</span></div>
        <div class="row"><span>Bet this street</span><span class="v" data-f="youBet">–</span></div>
      </div>
      <div class="sec"><div class="t" data-f="oppT">Opponent</div>
        <div class="row"><span>Position</span><span class="v" data-f="oppPos">–</span></div>
        <div class="row"><span>Stack</span><span class="v" data-f="oppStack">–</span></div>
        <div class="row"><span>Bet this street</span><span class="v" data-f="oppBet">–</span></div>
      </div>
      <div class="sec"><div class="t">Pot</div>
        <div class="row"><span>Pot (earlier streets)</span><span class="v" data-f="potMain">–</span></div>
        <div class="row"><span>Total pot (incl. bets)</span><span class="v" data-f="potTotal">–</span></div>
        <div class="row"><span>To call</span><span class="v" data-f="toCall">–</span></div>
        <div class="row"><span>Effective stack</span><span class="v" data-f="eff">–</span></div>
      </div>
      <div class="sec"><div class="t">Cards</div>
        <div class="row"><span>Yours</span><span class="v cards" data-f="hole">–</span></div>
        <div class="row"><span>Board</span><span class="v cards" data-f="board">–</span></div>
      </div>
      <div class="sec"><div class="t">Recommendation</div>
        <div data-f="status" class="mix">–</div>
        <div class="rec" data-f="rec"></div>
        <div class="mix" data-f="mix"></div>
        <div class="mix" data-f="src"></div>
      </div>
      <div class="sec row" style="align-items:center">
        <span>Your seat</span><select data-a="seat"><option value="">auto</option></select>
      </div>
      <div class="sec err" data-f="err" style="display:none"></div>
    </div>`;
  document.body.appendChild(root);
  const $ = (f) => root.querySelector(`[data-f="${f}"]`);
  const set = (f, v) => { const el = $(f); if (el) el.innerHTML = v; };

  // drag by the header
  (() => {
    const hd = root.querySelector(".hd"); let sx, sy, ox, oy;
    hd.addEventListener("mousedown", (e) => { if (e.target.tagName === "BUTTON") return;
      sx = e.clientX; sy = e.clientY; const r = root.getBoundingClientRect(); ox = r.left; oy = r.top;
      const mv = (ev) => { root.style.left = ox + ev.clientX - sx + "px"; root.style.top = oy + ev.clientY - sy + "px"; root.style.right = "auto"; };
      const up = () => { removeEventListener("mousemove", mv); removeEventListener("mouseup", up); };
      addEventListener("mousemove", mv); addEventListener("mouseup", up); });
  })();
  root.querySelector('[data-a="min"]').onclick = () => root.classList.toggle("collapsed");
  const seatSel = root.querySelector('[data-a="seat"]');
  seatSel.onchange = () => { state.seat = seatSel.value ? +seatSel.value : null; store.set("seat", state.seat); };

  // ------------------------------------------------------------------ formatting
  // Names, labels and notes come from the game log (other players choose their names): always escape.
  const esc = (x) => String(x).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const SUIT = { s: "♠", h: "♥", d: "♦", c: "♣" };
  const fmtCards = (cs) => (cs && cs.length) ? cs.filter((c) => /^[2-9TJQKA][shdc]$/.test(c)).map((c) => {
    const r = c[0] === "T" ? "10" : c[0]; const red = c[1] === "h" || c[1] === "d";
    return `<span class="${red ? "red" : ""}">${r}${SUIT[c[1]]}</span>`; }).join(" ") : "–";
  const chips = (c, b) => `${Number(c).toFixed(2)} <span class="bb">(${Number(b).toFixed(1)}bb)</span>`;
  const pctMix = (p) => Object.entries(p).filter(([, v]) => v >= 0.005).slice(0, 5)
    .map(([k, v]) => `${esc(k)} ${(v * 100).toFixed(0)}%`).join(" · ");

  // ------------------------------------------------------------------ reading the page
  const youName = () => {
    const el = document.querySelector(".you-player .table-player-name, .you-player [class*='name']");
    return el ? el.textContent.trim() : null;
  };

  async function fetchLog() {
    const urls = state.logUrl ? [state.logUrl] : [
      `/games/${gameId}/poker_now_log_${gameId}.csv`,
      `/games/${gameId}/log?after_at=&before_at=&mm=false`,
    ];
    for (const u of urls) {
      try {
        const r = await fetch(u, { credentials: "include", cache: "no-store" });
        if (!r.ok) continue;
        const text = await r.text();
        if (/starting hand|Player stacks|"logs"/.test(text) || text.length < 20) { state.logUrl = u; return text; }
      } catch {}
    }
    throw new Error("could not read the game log (tried the CSV and JSON log URLs)");
  }

  // ------------------------------------------------------------------ main loop
  function render(res) {
    const t = res.table || {};
    const h = t.hero, v = t.villain;
    set("hand", t.hand ? `#${t.hand}` : "–");
    set("street", esc(t.street || "–"));
    set("button", t.button ? `${esc(t.button)} ${t.hero_has_button ? "(you)" : "(opponent)"}` : "–");
    set("youT", h ? `You · seat #${esc(h.seat)} · ${esc(h.name)}` : "You");
    set("oppT", v ? `Opponent · seat #${esc(v.seat)} · ${esc(v.name)}` : "Opponent");
    set("youPos", h ? esc(h.position) : "–");
    set("oppPos", v ? esc(v.position) : "–");
    set("youStack", h ? chips(h.stack_now, h.stack_now_bb) : "–");
    set("oppStack", v ? chips(v.stack_now, v.stack_now_bb) : "–");
    set("youBet", h ? chips(h.bet_this_street, h.bet_this_street_bb) : "–");
    set("oppBet", v ? chips(v.bet_this_street, v.bet_this_street_bb) : "–");
    set("potMain", t.pot_main != null ? chips(t.pot_main, t.pot_main_bb) : "–");
    set("potTotal", t.pot_total != null ? chips(t.pot_total, t.pot_total_bb) : "–");
    set("toCall", t.to_call != null ? chips(t.to_call, t.to_call_bb) : "–");
    set("eff", t.effective_stack_bb != null ? `${t.effective_stack_bb}bb` : "–");
    set("hole", fmtCards(t.hero_cards));
    set("board", fmtCards(t.board));
    const d = res.decision;
    set("status", d ? "Your turn" : esc(res.wait || "–"));
    if (d) {
      const amt = (d.action === "bet" || d.action === "raise") ? ` to ${d.to_chips.toFixed(2)} (${d.to_bb}bb)` : "";
      set("rec", esc(`${d.action.toUpperCase()}${amt}`));
      set("mix", pctMix(d.probs));
      set("src", esc(`${d.source} · ${d.node}${d.notes && d.notes.length ? " · " + d.notes.join("; ") : ""}`));
    } else { set("rec", ""); set("mix", ""); set("src", ""); }
    const seats = t.seats_in_hand || [];
    const want = ["", ...seats.map((s) => String(s.seat))].join(",");
    if (seatSel.dataset.opts !== want) {
      seatSel.innerHTML = `<option value="">auto</option>` +
        seats.map((s) => `<option value="${esc(s.seat)}">#${esc(s.seat)} ${esc(s.name)}</option>`).join("");
      seatSel.dataset.opts = want;
    }
    seatSel.value = state.seat ? String(state.seat) : "";
  }

  async function tick() {
    if (state.busy) return;
    state.busy = true;
    const dot = root.querySelector(".dot"), err = $("err");
    try {
      const log = await fetchLog();
      const name = state.seat ? null : youName();
      const r = await fetch(`${SERVER}/decide`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ log, seat: state.seat, you_name: name }) });
      const res = await r.json();
      if (res.error) throw new Error(res.error);
      if (!state.seat && res.table && res.table.hero_seat) {      // lock the seat we were found in
        state.seat = res.table.hero_seat; store.set("seat", state.seat);
      }
      render(res);
      dot.style.background = "#2b9a5d"; err.style.display = "none";
    } catch (e) {
      dot.style.background = "#c0392b"; err.style.display = "block"; err.textContent = String(e.message || e);
    } finally { state.busy = false; }
  }

  const timer = setInterval(tick, POLL_MS);
  window.__pnbot = { timer, root, state, tick };
  tick();
})();
