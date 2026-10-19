// PokerNow page script for the match bot (experiments 31-32). Served by bot/pokernow/server.py.
// Reads the game log, asks the local server for a decision, shows a labeled overlay and, with auto-play
// on (the default), makes the move itself: a whole game with no clicks from the person running it.
// The hero is the player in the seat locked on first load.
(() => {
  const SERVER = (document.currentScript && new URL(document.currentScript.src).origin) || "http://127.0.0.1:8765";
  const TOKEN = "__PNBOT_TOKEN__";  // filled in by the server; it rejects requests without it
  const HEADERS = { "Content-Type": "application/json", "X-Pnbot-Token": TOKEN };
  const TURN_CHECK_MS = 350;        // how often the page is checked for "our turn" (cheap: no network)
  const REFRESH_MS = 2500;          // overlay refresh while waiting
  const RETRY_MS = 6000;            // an action the log hasn't recorded by then is retried with a safe fallback
  if (window.__pnbot) { clearInterval(window.__pnbot.timer); window.__pnbot.root?.remove(); }
  const gameId = (location.pathname.match(/\/games\/([^/?#]+)/) || [])[1];
  const KEY = (k) => `pnbot-${k}-${gameId}`;
  const store = {
    get: (k, d) => { try { const v = localStorage.getItem(KEY(k)); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set: (k, v) => { try { localStorage.setItem(KEY(k), JSON.stringify(v)); } catch {} },
  };
  const state = { seat: store.get("seat", null), auto: store.get("auto", true), busy: false, logUrl: null,
                  acted: {}, lastRefresh: 0, turnSince: 0 };

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
      #pnbot button.tg{font:inherit;background:#262c35;color:#e8e8e8;border:1px solid #39414c;border-radius:6px;
        padding:3px 8px;cursor:pointer}
      #pnbot button.tg.on{background:#1f6f43;border-color:#2b9a5d}
      #pnbot select{font:inherit;background:#262c35;color:#e8e8e8;border:1px solid #39414c;border-radius:6px;padding:2px 6px}
      #pnbot .min{border:none;background:none;padding:0 2px;color:#9aa4b2;font:inherit;cursor:pointer}
      #pnbot.collapsed .body{display:none}
      #pnbot .err{color:#ff8f8f;white-space:pre-wrap}
    </style>
    <div class="hd"><span class="dot"></span><b>Poker bot</b><button class="min" data-a="min">–</button></div>
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
      <div class="sec"><div class="t">Auto-play</div>
        <div class="mix" data-f="last">–</div>
      </div>
      <div class="sec row" style="align-items:center">
        <button class="tg" data-a="auto">Auto-play: on</button>
        <span>Seat <select data-a="seat"><option value="">auto</option></select></span>
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
  const autoBtn = root.querySelector('[data-a="auto"]');
  const paintAuto = () => { autoBtn.textContent = `Auto-play: ${state.auto ? "on" : "off"}`; autoBtn.classList.toggle("on", state.auto); };
  autoBtn.onclick = () => { state.auto = !state.auto; store.set("auto", state.auto); paintAuto(); };
  paintAuto();
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
    // The JSON log is what PokerNow's log panel loads; the CSV download sits behind a captcha.
    const urls = state.logUrl ? [state.logUrl] : [
      `/games/${gameId}/log?after_at=&before_at=&mm=false`,
      `/games/${gameId}/poker_now_log_${gameId}.csv`,
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

  // ------------------------------------------------------------------ acting
  // PokerNow marks the player whose decision it is with "decision-current". Its pre-action buttons
  // ("Check", "Fold", "Check or Fold") also show when it is NOT our turn, so buttons alone aren't enough.
  const ourTurnOnPage = () => !!document.querySelector(".you-player.decision-current");
  const visible = (el) => !!el && el.offsetParent !== null && !el.disabled;
  const findBtn = (re) => [...document.querySelectorAll("button")]
    .find((b) => visible(b) && !root.contains(b) && re.test(b.textContent.trim()));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function setAmount(input, value) {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  // One click on PokerNow's own buttons. Returns what was clicked, or throws if nothing fit.
  async function act(d) {
    const call = () => findBtn(/^call\b/i), check = () => findBtn(/^check$/i), fold = () => findBtn(/^fold$/i);
    const click = (b, what) => { if (!b) throw new Error(`no ${what} button`); b.click(); return what; };
    if (d.action === "fold") return check() ? click(check(), "check (free)") : click(fold(), "fold");
    if (d.action === "check") return check() ? click(check(), "check") : click(call() || fold(), "call/fold (no check)");
    if (d.action === "call") return call() ? click(call(), "call") : click(check(), "check");
    // bet / raise: "Bet" or "Raise" exactly (not the one-click "Bet 0.20"), then the amount, then confirm.
    const open = findBtn(/^(bet|raise)$/i);
    if (!open) return call() ? click(call(), "call (no raise possible)") : click(check(), "check (no bet possible)");
    open.click();
    await sleep(300);
    const form = document.querySelector(".raise-controller-form");
    if (!form) throw new Error("raise box did not open");
    const allIn = [...form.querySelectorAll("button")].find((b) => /^all in$/i.test(b.textContent.trim()));
    if (d.all_in && allIn) {
      allIn.click();
    } else {
      const input = form.querySelector("input.value");
      if (!input) throw new Error("no amount field in the raise box");
      setAmount(input, d.to_chips.toFixed(2));
    }
    await sleep(150);
    const submit = form.querySelector("input[type=submit], button[type=submit]");
    if (!submit) throw new Error("no confirm button in the raise box");
    submit.click();
    return `${d.action} to ${d.all_in ? "all-in" : d.to_chips.toFixed(2)}`;
  }

  async function report(r) {
    try { await fetch(`${SERVER}/acted`, { method: "POST", headers: HEADERS, body: JSON.stringify(r) }); } catch {}
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

  async function ask() {
    const log = await fetchLog();
    const name = state.seat ? null : youName();
    const r = await fetch(`${SERVER}/decide`, { method: "POST", headers: HEADERS,
      body: JSON.stringify({ log, seat: state.seat, you_name: name }) });
    const res = await r.json();
    if (res.error) throw new Error(res.error);
    if (!state.seat && res.table && res.table.hero_seat) {      // lock the seat we were found in
      state.seat = res.table.hero_seat; store.set("seat", state.seat);
    }
    render(res);
    return res;
  }

  async function tick() {
    if (state.busy) return;
    state.busy = true;
    const dot = root.querySelector(".dot"), err = $("err");
    try {
      const now = Date.now();
      const turn = ourTurnOnPage();
      if (!turn) state.turnSince = 0;
      else if (!state.turnSince) state.turnSince = now;
      // Back from "away" (e.g. after a timeout) without waiting for anyone.
      const back = findBtn(/^i'?m back$/i);
      if (back && state.auto) back.click();
      if (turn && state.auto) {
        const res = await ask();
        state.lastRefresh = Date.now();
        const d = res.decision;
        if (d) {
          const prev = state.acted[d.id];
          if (!prev) {
            const ms0 = Date.now() - state.turnSince;
            await sleep(Math.max(0, Math.min(700, 1500 - ms0)) * Math.random());   // a little human-like pause
            if (ourTurnOnPage()) {
              let did, error = null;
              try { did = await act(d); } catch (e) { error = String(e.message || e); }
              const ms = Date.now() - state.turnSince;
              state.acted[d.id] = { t: Date.now(), tries: 1 };
              set("last", esc(`hand ${res.table.hand}: ${did || "FAILED"} in ${(ms / 1000).toFixed(1)}s` + (error ? ` · ${error}` : "")));
              report({ id: d.id, action: did || d.action, ms, error });
            }
          } else if (Date.now() - prev.t > RETRY_MS && prev.tries < 3) {
            // The log never recorded our move: fall back to the safest action that commits no chips
            // (check if free, otherwise fold). The server re-confirmed above that it is still our turn.
            prev.tries += 1; prev.t = Date.now();
            let did, error = null;
            const safe = findBtn(/^check$/i) ? "check" : "fold";
            try { did = await act({ action: safe }); } catch (e) { error = String(e.message || e); }
            set("last", esc(`hand ${res.table.hand}: retry ${prev.tries - 1} → ${did || "FAILED"}` + (error ? ` · ${error}` : "")));
            report({ id: d.id, action: `retry: ${did}`, ms: Date.now() - state.turnSince, error: error || "action not recorded, fell back" });
          }
        }
      } else if (now - state.lastRefresh > REFRESH_MS) {
        state.lastRefresh = now;
        await ask();
      }
      dot.style.background = "#2b9a5d"; err.style.display = "none";
    } catch (e) {
      dot.style.background = "#c0392b"; err.style.display = "block"; err.textContent = String(e.message || e);
    } finally { state.busy = false; }
  }

  const timer = setInterval(tick, TURN_CHECK_MS);
  window.__pnbot = { timer, root, state, tick };
  tick();
})();
