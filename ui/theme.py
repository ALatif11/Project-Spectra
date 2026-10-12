"""Product theme: design tokens, injected CSS and small HTML helpers.

TOKENS is the single source of colors and fonts. The CSS below, the Plotly charts (ui/viz.py) and the
result meter read it; .streamlit/config.toml mirrors the base colors.
"""

import html

import streamlit as st

TOKENS = {
    "bg": "#0b0c0e", "surface": "#131418", "surface2": "#1a1b20", "border": "#25272d", "fg": "#ecedef",
    "muted": "#9a9ca6", "brand": "#5b8cff", "brand_soft": "rgba(91,140,255,.12)",
    # result semantics, separate from the brand color
    "human": "#4cc38a", "ai": "#f0616d", "unsure": "#e5a93a",
    "sans": "'Geist', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
    "mono": "'Geist Mono', ui-monospace, SFMono-Regular, Consolas, monospace",
}

_ROOT = ":root {" + "".join(f" --{k.replace('_', '-')}:{v};" for k, v in TOKENS.items()) + " --ease:cubic-bezier(.2,.8,.2,1); }"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');
/* Layout: slim product header, underline nav, then a two-column tool (input left, answer right). */
%ROOT%
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input { font-family: var(--sans); }
.stApp { background: var(--bg); color: var(--fg); }
html, body, .stApp, [data-testid="stMain"] { overflow-x: clip; }
header[data-testid="stHeader"], #MainMenu, footer { display:none; }
.block-container { max-width: 1120px; padding: 1.25rem 1.25rem 5rem; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
@keyframes fade { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform:none; } }
@keyframes fill { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes pop { from { left:0; } }

/* Header */
.topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.4rem 0 1.1rem; }
.logo { display:flex; align-items:center; gap:.55rem; font-weight:600; font-size:1.02rem; letter-spacing:-.01em; }
.logo svg { display:block; }
.privacy { display:inline-flex; align-items:center; gap:.45rem; font-size:.8rem; color: var(--muted); padding:.3rem .7rem;
  border:1px solid var(--border); border-radius:999px; }
.privacy i { width:7px; height:7px; border-radius:50%; background: var(--human); }

/* Nav: Streamlit tabs as an underline nav */
.stTabs [role="tablist"] { gap:1.4rem; border-bottom:1px solid var(--border); margin-bottom:2rem; overflow-x:auto; }
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before { display:none !important; }
.stTabs [data-testid="stTab"] { height:auto; min-height:42px; padding:.4rem 0 .7rem; color: var(--muted); background:none; transition: color .2s var(--ease); }
.stTabs [data-testid="stTab"] p { font-size:.92rem; font-weight:500; }
.stTabs [data-testid="stTab"]:hover, .stTabs [data-testid="stTab"][aria-selected="true"] { color: var(--fg); }
.stTabs .react-aria-SelectionIndicator { height:2px !important; background: var(--fg) !important; border:none !important; bottom:-1px !important;
  transition-timing-function: var(--ease) !important; }

/* Page heading */
.page-h { margin: 0 0 1.6rem; max-width: 40rem; }
.page-h h1 { font-size: clamp(1.6rem, 3vw, 2.1rem); font-weight:600; letter-spacing:-.025em; line-height:1.2; margin:0 0 .5rem; color: var(--fg); }
.page-h p { margin:0; color: var(--muted); line-height:1.6; }

/* Cards: plain surfaces */
[class*="st-key-card"] { padding:1.3rem 1.4rem; border:1px solid var(--border); border-radius:14px; background: var(--surface);
  animation: fade .3s var(--ease) both; }
.card-h { display:flex; align-items:center; gap:.45rem; margin:0 0 .9rem; font-size:.78rem; font-weight:500; color: var(--muted);
  text-transform:uppercase; letter-spacing:.06em; }
.card-h .n { color: var(--fg); }
@media (max-width: 760px) { [class*="st-key-card"] { padding:1.1rem 1rem; } .block-container { padding:1rem 1rem 4rem; } }

/* Hover explanations */
[data-tip] { position:relative; cursor:help; }
[data-tip]::after { content: attr(data-tip); position:absolute; left:-.4rem; bottom: calc(100% + 8px); z-index:20;
  width:max-content; max-width: min(18rem, 80vw); padding:.55rem .7rem; border-radius:8px; background: var(--fg); color: var(--bg);
  font: 400 12.5px/1.5 var(--sans); letter-spacing:0; text-transform:none; text-align:left; white-space:normal;
  box-shadow: 0 10px 30px -10px rgba(0,0,0,.6); opacity:0; transform: translateY(3px); pointer-events:none;
  transition: opacity .15s var(--ease), transform .15s var(--ease); }
[data-tip].end::after { left:auto; right:-.4rem; }
[data-tip].below::after { bottom:auto; top: calc(100% + 8px); }
[data-tip]:hover::after, [data-tip]:focus-visible::after { opacity:1; transform:none; }
[data-tip]:focus-visible, .stApp button:focus-visible, .stApp a:focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; border-radius:6px; }
.i { display:inline-grid; place-items:center; width:16px; height:16px; border-radius:50%; border:1px solid var(--border);
  font: 500 10px var(--sans); color: var(--muted); text-transform:none; letter-spacing:0; vertical-align:middle; }
.i:hover, .i:focus-visible { color: var(--fg); border-color: var(--muted); }
.term { border-bottom: 1px dotted var(--muted); }

/* Text bits and chips */
.note { color: var(--muted); font-size:.86rem; line-height:1.6; margin:.45rem 0 0; }
.note.err { color: var(--ai); } .note.mono { font-family: var(--mono); font-size:.78rem; }
.chip { display:inline-flex; align-items:center; gap:.35rem; padding:.2rem .6rem; border-radius:999px; font-size:.75rem; font-weight:500;
  background: var(--surface2); color: var(--muted); margin:0 .3rem .3rem 0; border:1px solid var(--border); }
.chip.ok { color: var(--human); } .chip.bad { color: var(--ai); } .chip.warn { color: var(--unsure); }
.chip::before { content:""; width:6px; height:6px; border-radius:50%; background: currentColor; } .chip.plain::before { display:none; }

/* Result */
.answer { display:flex; gap:.9rem; align-items:flex-start; margin:.1rem 0 1rem; }
.answer .dot { flex:none; width:12px; height:12px; border-radius:50%; margin-top:.55rem; }
.answer h2 { font-size:1.35rem; font-weight:600; letter-spacing:-.015em; line-height:1.3; margin:0; color: var(--fg); }
.answer p { margin:.3rem 0 0; color: var(--muted); font-size:.9rem; line-height:1.55; }
.answer.ai .dot { background: var(--ai); box-shadow: 0 0 0 4px rgba(240,97,109,.15); }
.answer.human .dot { background: var(--human); box-shadow: 0 0 0 4px rgba(76,195,138,.15); }
.answer.unsure .dot { background: var(--unsure); box-shadow: 0 0 0 4px rgba(229,169,58,.15); }
.answer.off .dot { background: var(--muted); }
.meter { margin: .4rem 0 .2rem; }
.meter .track { position:relative; height:8px; border-radius:999px;
  background: linear-gradient(90deg, rgba(76,195,138,.35), rgba(229,169,58,.35) 50%, rgba(240,97,109,.35)); }
.meter .band { position:absolute; top:-3px; bottom:-3px; border-radius:4px; border:1px dashed rgba(229,169,58,.7); }
.meter .thr { position:absolute; top:-6px; bottom:-6px; width:2px; background: var(--fg); opacity:.55; }
.meter .pin { position:absolute; top:50%; width:16px; height:16px; margin:-8px 0 0 -8px; border-radius:50%; background: var(--fg);
  border:3px solid var(--bg); box-shadow: 0 0 0 1px var(--fg); animation: pop .7s var(--ease) both; }
.meter .ends { display:flex; justify-content:space-between; margin-top:.5rem; font-size:.75rem; color: var(--muted); }
.figures { display:flex; flex-wrap:wrap; gap:1.2rem; margin:.8rem 0 0; font-size:.8rem; color: var(--muted); }
.figures b { font-family: var(--mono); font-weight:500; color: var(--fg); margin-left:.3rem; font-variant-numeric: tabular-nums; }
.section-h { margin:1.3rem 0 .6rem; font-size:.85rem; font-weight:600; color: var(--fg); display:flex; align-items:center; gap:.4rem; }
.why { list-style:none; padding:0; margin:0; display:grid; gap:.55rem; }
.why li { display:grid; grid-template-columns: 1fr auto; gap:.2rem .8rem; align-items:center; padding:.6rem .75rem; border-radius:10px; background: var(--surface2); }
.why .what { font-size:.88rem; color: var(--fg); } .why .lvl { font-size:.78rem; color: var(--muted); }
.why .dir { font-size:.75rem; font-weight:500; white-space:nowrap; } .why .dir.ai { color: var(--ai); } .why .dir.human { color: var(--human); }
.why .bar { grid-column: 1 / -1; height:3px; border-radius:999px; background: var(--border); overflow:hidden; }
.why .bar b { display:block; height:100%; transform-origin:left; animation: fill .6s var(--ease) both; }
.why .ai .bar b, .why li.ai .bar b { background: var(--ai); } .why li.human .bar b { background: var(--human); }
.warns { list-style:none; padding:0; margin:0; display:grid; gap:.4rem; }
.warns li { font-size:.85rem; color: var(--muted); line-height:1.5; padding-left:1.1rem; position:relative; }
.warns li::before { content:"!"; position:absolute; left:0; top:0; font-weight:700; color: var(--unsure); }
.fine { margin-top:1.1rem; padding-top:.9rem; border-top:1px solid var(--border); font-size:.78rem; color: var(--muted); line-height:1.55; }

/* Next steps */
.steps-list { list-style:none; padding:0; margin:0; display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:.7rem; counter-reset: s; }
.steps-list li { counter-increment: s; padding:.85rem .9rem .85rem 2.6rem; border-radius:10px; background: var(--surface2); position:relative; }
.steps-list li::before { content: counter(s); position:absolute; left:.85rem; top:.85rem; width:1.2rem; height:1.2rem; border-radius:50%;
  display:grid; place-items:center; font-size:.72rem; font-weight:600; background: var(--brand-soft); color: var(--brand); }
.steps-list b { display:block; font-size:.9rem; font-weight:600; margin-bottom:.15rem; } .steps-list span { font-size:.82rem; color: var(--muted); line-height:1.5; }
.links { margin-top:.9rem; font-size:.84rem; color: var(--muted); } .links a { color: var(--brand); text-decoration:none; margin-right:1rem; }
.links a:hover { text-decoration:underline; }
@media (max-width: 760px) { .steps-list { grid-template-columns: 1fr; } }

/* Empty / before upload */
.tips { list-style:none; padding:0; margin:.2rem 0 0; display:grid; gap:.7rem; }
.tips li { display:flex; gap:.7rem; font-size:.88rem; color: var(--muted); line-height:1.5; }
.tips li b { color: var(--fg); font-weight:500; }
.tips li::before { content:""; flex:none; width:6px; height:6px; margin-top:.5rem; border-radius:50%; background: var(--brand); }

/* Widgets */
[data-testid="stFileUploaderDropzone"] { border:1px dashed #3a3d45; border-radius:12px; background: var(--bg); padding:1.6rem; transition: border-color .2s var(--ease), background .2s var(--ease); }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--brand); background: rgba(91,140,255,.04); }
.stApp [data-testid="stBaseButton-secondary"], .stApp [data-testid="stBaseButton-primary"], [data-testid="stFileUploaderDropzone"] button,
.stApp [data-testid="stDownloadButton"] button { border-radius:8px; font-weight:500; min-height:40px; }
.stApp [data-testid="stBaseButton-primary"] { background: var(--brand); border-color: var(--brand); color:#fff; }
[data-testid="stExpander"] details { border-radius:12px; border-color: var(--border); background: var(--surface); }
audio { width:100%; }

/* Model page */
.agrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap:.7rem; }
.acard { padding:1rem; border-radius:10px; background: var(--surface2); min-width:0; }
.acard h4 { font-size:.92rem; font-weight:600; margin:0 0 .55rem; color: var(--fg); }
.kv { display:grid; grid-template-columns:auto 1fr; gap:.25rem .9rem; margin:.6rem 0; font-size:.82rem; }
.kv dt { color: var(--muted); } .kv dd { margin:0; font-family: var(--mono); overflow-wrap:anywhere; }
.stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(7.5rem,1fr)); gap:.5rem; margin:.6rem 0; }
.stat { padding:.6rem .75rem; border-radius:8px; background: var(--surface); border:1px solid var(--border); }
.stat b { display:block; font: 500 1.1rem var(--mono); font-variant-numeric: tabular-nums; }
.stat span { font-size:.75rem; color: var(--muted); }
.big-stats .stat b { font-size:1.35rem; }
.bars { display:grid; gap:.35rem; }
.bar { display:grid; grid-template-columns: 5.5rem 1fr 6.2rem; gap:.6rem; align-items:center; font-size:.8rem; }
.bar em { font: normal 500 .8rem var(--mono); text-align:right; font-variant-numeric: tabular-nums; }
.empty { text-align:center; padding:2.4rem 1rem; } .empty h3 { font-size:1.1rem; font-weight:600; margin:0 0 .4rem; }
.empty p { color: var(--muted); max-width:30rem; margin:0 auto; line-height:1.6; }

/* Tables and flow diagrams */
.tbl { overflow-x:auto; } table.cmp { width:100%; min-width:28rem; border-collapse:collapse; font-size:.86rem; }
table.cmp th { text-align:left; font-size:.72rem; font-weight:500; text-transform:uppercase; letter-spacing:.06em; color: var(--muted); padding:.45rem .6rem; }
table.cmp td { padding:.55rem .6rem; border-top:1px solid var(--border); font-variant-numeric: tabular-nums; }
.flow { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem; margin:.2rem 0 .4rem; }
.flow .node { padding:.5rem .75rem; border-radius:8px; background: var(--surface2); border:1px solid var(--border); font-size:.84rem; }
.flow .arrow { color: var(--muted); }
/* How it works */
.prose { max-width: 44rem; } .prose h3 { font-size:1rem; font-weight:600; margin:1.6rem 0 .4rem; }
.prose p, .prose li { color: var(--muted); line-height:1.65; font-size:.92rem; }

@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
</style>
""".replace("%ROOT%", _ROOT)

_MARK = ('<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><rect width="22" height="22" rx="6" fill="#5b8cff"/>'
         '<path d="M5 11h1.5M8 7.5v7M11 5v12M14 8v6M17 11h0" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>')


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header() -> None:
    st.markdown(f'<div class="topbar"><div class="logo">{_MARK}EchoProof</div>'
                '<span class="privacy" data-tip="Audio is decoded and analyzed in memory on this computer. Nothing is '
                'sent to a server." tabindex="0"><i></i>Runs on this device</span></div>', unsafe_allow_html=True)


def page_heading(title: str, body: str) -> None:
    st.markdown(f'<div class="page-h"><h1>{html.escape(title)}</h1><p>{html.escape(body)}</p></div>', unsafe_allow_html=True)


def tip(text: str, cls: str = "") -> str:
    """A small hover/focus explanation dot."""
    t = html.escape(text, quote=True)
    return f'<span class="i {cls}" tabindex="0" role="img" aria-label="{t}" data-tip="{t}">i</span>'


def term(label: str, text: str, cls: str = "") -> str:
    """Inline words with a dotted underline that explain themselves on hover."""
    t = html.escape(text, quote=True)
    return f'<span class="term {cls}" tabindex="0" aria-label="{html.escape(label)}: {t}" data-tip="{t}">{html.escape(label)}</span>'


def card_title(text: str, explain: str | None = None, n: str | None = None) -> None:
    num = f'<span class="n">{html.escape(n)}</span>' if n else ""
    st.markdown(f'<div class="card-h">{num}{html.escape(text)}{tip(explain) if explain else ""}</div>',
                unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',
                unsafe_allow_html=True)
