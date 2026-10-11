"""Studio-paper theme: injected CSS plus small HTML helpers (cards, hover explanations)."""

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');
/* Layout: one calm column of paper cards; the result gauge and the spectrogram "screen" are the two focal objects. */
:root {
  --paper:#0D0E11; --card:#15161B; --tile:#1C1D23; --tile2:#22232A; --track:#1A1B21; --ink:#ECEBF0; --mute:#8E8D98;
  --hair:rgba(255,255,255,.08); --signal:#7B88FF; --human:#3CCB9A; --synth:#FF6B6F; --amber:#E9B44C; --screen:#08090C;
  --tip-bg:#F2F1ED; --tip-fg:#141418;
  --display:'Bricolage Grotesque', 'Geist', ui-sans-serif, system-ui, sans-serif;
  --body:'Geist', ui-sans-serif, system-ui, sans-serif; --mono:'Geist Mono', ui-monospace, monospace;
  --ease:cubic-bezier(0.32, 0.72, 0, 1); --spring:cubic-bezier(0.34, 1.4, 0.64, 1);
}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp button { font-family: var(--body); }
.stApp { background: var(--paper); color: var(--ink); }
html, body, .stApp, [data-testid="stMain"] { overflow-x: clip; }
header[data-testid="stHeader"], #MainMenu, footer { display:none; }
.block-container { max-width: 1180px; padding: 3rem 1.25rem 6rem; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }

@keyframes rise { from { opacity:0; transform: translateY(18px); } to { opacity:1; transform:none; } }
@keyframes sweep { from { transform: rotate(-90deg); } to { transform: rotate(var(--a)); } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* Hero */
.hero { display:grid; grid-template-columns: minmax(0,1.4fr) minmax(0,1fr); gap:2.5rem; align-items:end; margin:0 0 2.6rem; animation: rise .9s var(--ease) both; }
.hero h1 { font-family: var(--display); font-weight:700; font-size: clamp(2.6rem, 6vw, 4.8rem); line-height:.98; letter-spacing:-.035em; margin:0; color:var(--ink); text-wrap:balance; }
.hero h1 em { font-style:normal; color:var(--signal); }
.hero p { margin:0; color:var(--mute); font-size:1.02rem; line-height:1.6; max-width:30rem; }
.hero .brand { display:flex; align-items:center; gap:.55rem; font: 500 12px var(--mono); letter-spacing:.06em; color:var(--ink); margin-bottom:1.4rem; }
.hero .brand i { width:22px; height:14px; display:inline-flex; gap:2px; align-items:center; }
.hero .brand i b { width:2px; border-radius:2px; background:var(--signal); }
@media (max-width: 760px) { .hero { grid-template-columns: 1fr; gap:1rem; } }

/* Tabs as a segmented control; react-aria's sliding SelectionIndicator becomes the moving pill */
.stTabs [role="tablist"] { gap:.2rem; width:max-content; max-width:100%; overflow-x:auto; padding:.3rem; border-radius:999px;
  background:var(--track); box-shadow: inset 0 1px 2px rgba(0,0,0,.4); margin-bottom:2rem; }
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before { display:none !important; }
.stTabs [data-testid="stTab"] { position:relative; height:auto; padding:.55rem 1.15rem; border-radius:999px; color:var(--mute);
  transition: color .5s var(--ease), transform .4s var(--ease); }
.stTabs [data-testid="stTab"] p { position:relative; z-index:1; font-weight:500; font-size:.92rem; }
.stTabs [data-testid="stTab"]:hover { color:var(--ink); }
.stTabs [data-testid="stTab"]:active { transform: scale(.97); }
.stTabs [data-testid="stTab"][aria-selected="true"] { color:var(--ink); }
.stTabs .react-aria-SelectionIndicator { position:absolute !important; inset:0 !important; height:auto !important; width:auto !important;
  border-radius:999px; background:var(--tile2) !important; border:none !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.35), 0 6px 16px -8px rgba(0,0,0,.45); transition-timing-function: var(--ease) !important; }

/* Cards: keyed container is the inner core; layered shadows draw the outer tray */
[class*="st-key-card"] { margin:.45rem; padding:1.5rem 1.6rem 1.4rem; border-radius:1.6rem; background:var(--card);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 0 0 .45rem rgba(255,255,255,.02), 0 0 0 calc(.45rem + 1px) var(--hair),
              0 30px 60px -36px rgba(0,0,0,.6);
  animation: rise .8s var(--ease) both; transition: transform .7s var(--ease), box-shadow .7s var(--ease); }
[class*="st-key-card"]:hover { transform: translateY(-2px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 0 0 .45rem rgba(255,255,255,.03), 0 0 0 calc(.45rem + 1px) rgba(255,255,255,.15),
              0 36px 70px -36px rgba(0,0,0,.7); }
[class*="st-key-card"]:nth-of-type(2) { animation-delay:.08s } [class*="st-key-card"]:nth-of-type(3) { animation-delay:.16s }
.card-h { display:flex; align-items:center; gap:.5rem; margin:0 0 1rem; font: 600 12px var(--body); letter-spacing:.08em; text-transform:uppercase; color:var(--mute); }
.card-h .n { font-family: var(--mono); color:var(--signal); letter-spacing:0; }
@media (max-width: 760px) { [class*="st-key-card"] { padding:1.2rem 1rem; } .block-container { padding:1.6rem 1rem 4rem; } }

/* Hover explanations: any element with data-tip */
[data-tip] { position:relative; cursor:help; }
[data-tip]::after { content: attr(data-tip); position:absolute; left:-.4rem; bottom: calc(100% + 10px); z-index:20;
  width:max-content; max-width: min(19rem, 80vw); padding:.65rem .8rem; border-radius:.8rem; background:var(--tip-bg); color:var(--tip-fg);
  font: 400 12.5px/1.5 var(--body); letter-spacing:0; text-transform:none; text-align:left; white-space:normal;
  box-shadow: 0 18px 40px -16px rgba(0,0,0,.6); opacity:0; transform: translateY(6px) scale(.98); transform-origin: bottom left;
  pointer-events:none; transition: opacity .35s var(--ease), transform .45s var(--spring); }
[data-tip].end::after { left:auto; right:-.4rem; transform-origin: bottom right; }
[data-tip].below::after { bottom:auto; top: calc(100% + 10px); transform-origin: top left; }
[data-tip]:hover::after, [data-tip]:focus-visible::after { opacity:1; transform:none; }
[data-tip]:focus-visible { outline: 2px solid var(--signal); outline-offset: 3px; border-radius:6px; }
.i { display:inline-grid; place-items:center; width:17px; height:17px; border-radius:50%; border:1px solid rgba(255,255,255,.2);
  font: 500 10px var(--mono); color:var(--mute); text-transform:none; letter-spacing:0; vertical-align:middle;
  transition: background .4s var(--ease), color .4s var(--ease), border-color .4s var(--ease); }
.i:hover, .i:focus-visible { background:var(--ink); color:var(--paper); border-color:var(--ink); }
.term { border-bottom: 1px dashed rgba(255,255,255,.35); }

/* Text bits */
.note { color:var(--mute); font-size:.88rem; line-height:1.6; margin:.4rem 0 0; }
.note.subtle { font-size:.8rem; padding-left:.65rem; border-left:2px solid rgba(233,180,76,.45); }
.note.err { color:var(--synth); } .note.mono { font-family: var(--mono); font-size:.78rem; }
.chip { display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .7rem; border-radius:999px; font: 500 11.5px var(--body);
  background:var(--tile); color:var(--ink); margin:0 .35rem .35rem 0; }
.chip::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; opacity:.55; }
.chip.ok { background:rgba(60,203,154,.12); color:#6FE3BA; } .chip.bad { background:rgba(255,107,111,.12); color:#FF9A9C; }
.chip.warn { background:rgba(233,180,76,.12); color:#F2C877; } .chip.plain::before { display:none; }

/* Detector result */
.verdict { font-family: var(--display); font-weight:700; font-size: clamp(1.6rem, 3vw, 2.15rem); letter-spacing:-.025em; line-height:1.05; margin:.6rem 0 .3rem; }
.verdict.human { color:var(--human); } .verdict.synth { color:var(--synth); } .verdict.mid { color:var(--amber); } .verdict.off { color:var(--mute); }
.gauge { display:block; width:100%; max-width:300px; margin:.4rem auto .2rem; overflow:visible; }
.gauge .needle { transform-box: view-box; transform-origin: 150px 150px; animation: sweep 1.4s var(--spring) .15s both; transform: rotate(var(--a)); }
.gauge text { font: 500 11px var(--mono); fill: var(--mute); }
.readout { display:grid; grid-template-columns: 1fr 1fr; gap:.6rem; margin:.6rem 0 .2rem; }
.readout div { padding:.75rem .9rem; border-radius:1rem; background:var(--tile); }
.readout b { display:block; font: 500 1.35rem var(--mono); color:var(--ink); font-variant-numeric: tabular-nums; }
.readout span { font-size:11.5px; color:var(--mute); }
.note.dial { text-align:center; margin:0 0 .4rem; font-size:.8rem; }
.model-tag { display:flex; flex-wrap:wrap; gap:.3rem; margin-top:1rem; }

/* How it works */
.steps { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:.8rem; margin-top:1.2rem; }
.step { padding:1rem; border-radius:1.1rem; background:var(--tile); animation: rise .8s var(--ease) both; }
.step:nth-child(2) { animation-delay:.1s } .step:nth-child(3) { animation-delay:.2s }
.step .n { font: 500 11px var(--mono); color:var(--signal); }
.step b { display:block; font-family: var(--display); font-size:1.05rem; margin:.25rem 0 .2rem; }
.step p { margin:0; font-size:.84rem; color:var(--mute); line-height:1.5; }
@media (max-width: 760px) { .steps { grid-template-columns: 1fr; } }

/* Uploader */
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed rgba(255,255,255,.2); border-radius:1.3rem; background:var(--tile); padding:1.6rem;
  transition: border-color .5s var(--ease), background .5s var(--ease), transform .5s var(--ease); }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--signal); background:rgba(123,136,255,.07); transform: scale(1.005); }
[data-testid="stFileUploaderDropzone"] button { border-radius:999px; background:var(--ink); color:var(--paper); border:none; padding:.5rem 1.1rem;
  transition: transform .4s var(--ease); }
[data-testid="stFileUploaderDropzone"] button:hover { transform: translateY(-1px); } [data-testid="stFileUploaderDropzone"] button:active { transform: scale(.97); }
audio { width:100%; border-radius:999px; }

/* Lab */
.agrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap:.8rem; }
.acard { padding:1.1rem 1.1rem 1rem; border-radius:1.2rem; background:var(--tile); min-width:0;
  transition: transform .6s var(--ease), background .6s var(--ease); animation: rise .8s var(--ease) both; }
.acard:hover { transform: translateY(-3px); background:var(--tile2); }
.acard:nth-child(2) { animation-delay:.08s } .acard:nth-child(3) { animation-delay:.16s }
.acard h4 { font-family: var(--display); font-size:1.05rem; margin:0 0 .6rem; color:var(--ink); }
.kv { display:grid; grid-template-columns:auto 1fr; gap:.25rem .9rem; margin:.7rem 0; font-size:.84rem; }
.kv dt { color:var(--mute); } .kv dd { margin:0; font-family: var(--mono); overflow-wrap:anywhere; }
.stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(7.5rem,1fr)); gap:.5rem; margin:.7rem 0; }
.stat { padding:.65rem .8rem; border-radius:.9rem; background:var(--card); }
.stat b { display:block; font: 500 1.15rem var(--mono); font-variant-numeric: tabular-nums; }
.stat span { font-size:11.5px; color:var(--mute); }
.big-stats .stat { background:var(--tile); } .big-stats .stat b { font-size:1.5rem; }
.intro { max-width:44rem; color:var(--mute); line-height:1.6; margin:0 0 1.2rem; }

/* Validation vs final test */
.verdict-sm { font-family: var(--display); font-size:1.2rem; font-weight:500; line-height:1.4; margin:0 0 1.1rem; max-width:46rem; }
.exam { display:grid; gap:.9rem; }
.exam-row { display:grid; grid-template-columns: minmax(9rem, 13rem) 1fr; gap:1rem; align-items:center; }
.exam-row .lbl b { display:block; font-weight:600; } .exam-row .lbl span { font-size:.8rem; color:var(--mute); }
.bars { display:grid; gap:.35rem; }
.bar { display:grid; grid-template-columns: 5.5rem 1fr 6rem; gap:.6rem; align-items:center; font-size:.82rem; }
.bar i { display:block; height:10px; border-radius:999px; background:var(--track); overflow:hidden; }
.bar i b { display:block; height:100%; border-radius:999px; transform-origin:left; animation: grow 1.1s var(--ease) .2s both; }
.bar.base i b { background:#5E5D68; } .bar.opt i b { background:var(--signal); }
.bar em { font: normal 500 .82rem var(--mono); text-align:right; font-variant-numeric: tabular-nums; }
.delta { font: 500 .78rem var(--mono); } .delta.up { color:var(--human); } .delta.down { color:var(--synth); }
@media (max-width: 760px) { .exam-row { grid-template-columns: 1fr; } }

/* Empty */
.empty { text-align:center; padding:3rem 1rem; }
.empty h3 { font-family: var(--display); font-size:1.5rem; margin:.4rem 0 .5rem; }
.empty p { color:var(--mute); max-width:30rem; margin:0 auto; line-height:1.6; }

@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def tip(text: str, cls: str = "") -> str:
    """A small hover/focus explanation dot."""
    t = html.escape(text, quote=True)
    return f'<span class="i {cls}" tabindex="0" role="img" aria-label="{t}" data-tip="{t}">i</span>'


def term(label: str, text: str, cls: str = "") -> str:
    """Inline words with a dashed underline that explain themselves on hover."""
    t = html.escape(text, quote=True)
    return f'<span class="term {cls}" tabindex="0" aria-label="{html.escape(label)}: {t}" data-tip="{t}">{html.escape(label)}</span>'


def hero() -> None:
    bars = "".join(f'<b style="height:{h}px"></b>' for h in (5, 11, 7, 14, 9, 4))
    st.markdown(
        f'<div class="hero"><div><div class="brand"><i>{bars}</i>ECHOPROOF</div>'
        "<h1>Is that voice <em>real</em>, or made by a machine?</h1></div>"
        "<p>Drop in a recording. EchoProof listens to it on this computer and scores how closely it resembles the "
        "AI-generated speech it has studied. It is a research tool: a hint, not a verdict.</p></div>",
        unsafe_allow_html=True,
    )


def card_title(text: str, explain: str | None = None, n: str | None = None) -> None:
    num = f'<span class="n">{html.escape(n)}</span>' if n else ""
    st.markdown(f'<div class="card-h">{num}{html.escape(text)}{tip(explain) if explain else ""}</div>',
                unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',
                unsafe_allow_html=True)
