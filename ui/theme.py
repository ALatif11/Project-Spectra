"""Ethereal Glass theme: injected CSS and small HTML helpers."""

import streamlit as st

EASE = "cubic-bezier(0.32, 0.72, 0, 1)"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');
:root { --ease: cubic-bezier(0.32, 0.72, 0, 1); --hair: rgba(255,255,255,.09); --mint:#5eead4; --violet:#a78bfa; --mute:#8b8b95; }
html, body, [class*="st-"], .stApp { font-family: 'Geist', ui-sans-serif, system-ui, sans-serif; }
.stApp { background:
  radial-gradient(60rem 40rem at 8% -10%, rgba(94,234,212,.10), transparent 60%),
  radial-gradient(50rem 36rem at 95% 0%, rgba(167,139,250,.12), transparent 60%),
  #050505; }
header[data-testid="stHeader"], #MainMenu, footer { display:none; }
.block-container { max-width: 1240px; padding: 2.5rem 1.25rem 6rem; }

@keyframes rise { from { opacity:0; transform: translateY(28px); filter: blur(10px); } to { opacity:1; transform:none; filter:none; } }
.rise { animation: rise .9s var(--ease) both; }
.rise.d1 { animation-delay:.08s } .rise.d2 { animation-delay:.16s } .rise.d3 { animation-delay:.24s }

/* Hero */
.eyebrow { display:inline-block; padding:.3rem .75rem; border:1px solid var(--hair); border-radius:999px;
  background:rgba(255,255,255,.04); font-size:10px; letter-spacing:.2em; text-transform:uppercase; color:var(--mint); font-weight:500; }
.hero h1 { font-size: clamp(2.4rem, 6vw, 4.6rem); line-height:1.02; letter-spacing:-.035em; font-weight:600; margin:1.1rem 0 .9rem; color:#f4f4f6 !important; }
.hero h1 span { background:linear-gradient(100deg,var(--mint),var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { max-width: 38rem; color:var(--mute); font-size:1.05rem; line-height:1.6; margin:0 0 2.2rem; }

/* Tabs as a floating pill */
div[data-baseweb="tab-list"] { gap:.25rem; width:max-content; max-width:100%; padding:.3rem; border:1px solid var(--hair);
  border-radius:999px; background:rgba(255,255,255,.04); margin-bottom:2rem; overflow-x:auto; }
button[data-baseweb="tab"] { border-radius:999px; padding:.55rem 1.1rem; height:auto; color:var(--mute);
  transition: all .6s var(--ease); background:transparent; }
button[data-baseweb="tab"][aria-selected="true"] { background:rgba(255,255,255,.10); color:#fff; box-shadow: inset 0 1px 1px rgba(255,255,255,.15); }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display:none; }

/* Double-bezel: the keyed container is the inner core; box-shadow rings form the outer shell */
[class*="st-key-bezel"] { margin:.4rem; padding:1.6rem 1.6rem 1.4rem; border-radius:calc(2rem - .4rem);
  background:linear-gradient(180deg,#0e0e11,#09090b);
  box-shadow: inset 0 1px 1px rgba(255,255,255,.12), 0 0 0 .4rem rgba(255,255,255,.03), 0 0 0 calc(.4rem + 1px) var(--hair);
  animation: rise .9s var(--ease) both; transition: box-shadow .7s var(--ease); }
[class*="st-key-bezel"]:hover { box-shadow: inset 0 1px 1px rgba(255,255,255,.12), 0 0 0 .4rem rgba(255,255,255,.04), 0 0 0 calc(.4rem + 1px) rgba(255,255,255,.18); }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
.card-h { display:flex; align-items:center; gap:.6rem; margin:0 0 1rem; font-size:10px; letter-spacing:.2em; text-transform:uppercase; color:var(--mute); font-weight:500; }
.card-h i { width:6px; height:6px; border-radius:50%; background:var(--mint); box-shadow:0 0 12px var(--mint); }

/* Result */
.verdict { font-size: clamp(1.6rem,3vw,2.2rem); font-weight:600; letter-spacing:-.02em; margin:.2rem 0 .5rem; color:#f4f4f6; }
.verdict.off { color:#a1a1aa; }
.note { color:var(--mute); font-size:.9rem; line-height:1.55; }
.stats { display:grid; grid-template-columns:1fr 1fr; gap:.6rem; margin:1.1rem 0; }
.stat { padding:.8rem 1rem; border:1px solid var(--hair); border-radius:1.1rem; background:rgba(255,255,255,.025); }
.stat b { display:block; font-family:'Geist Mono',monospace; font-size:1.25rem; font-weight:500; color:#f4f4f6; }
.stat span { font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:var(--mute); }
.pill { display:inline-block; padding:.25rem .7rem; border-radius:999px; font-size:11px; border:1px solid var(--hair); color:var(--mute); margin-right:.4rem; }
.pill.warn { color:#fcd34d; border-color:rgba(252,211,77,.3); }

.meter { position:relative; height:6px; border-radius:999px; background:rgba(255,255,255,.08); margin:.4rem 0 1rem; }
.meter b { position:absolute; inset:0 auto 0 0; border-radius:999px; background:linear-gradient(90deg,var(--mint),var(--violet)); transition: width .8s var(--ease); }
.meter i { position:absolute; top:-5px; bottom:-5px; width:2px; background:#fff; border-radius:2px; }
/* Uploader */
[data-testid="stFileUploaderDropzone"] { border:1px dashed rgba(255,255,255,.16); border-radius:1.4rem; background:rgba(255,255,255,.02);
  padding:1.8rem; transition: all .6s var(--ease); }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--mint); background:rgba(94,234,212,.04); }
[data-testid="stFileUploaderDropzone"] button { border-radius:999px; border:1px solid var(--hair); background:rgba(255,255,255,.07); color:#fff; transition: transform .5s var(--ease); }
[data-testid="stFileUploaderDropzone"] button:active { transform: scale(.98); }
audio { width:100%; filter: invert(.92) hue-rotate(140deg) saturate(.6); border-radius:999px; }

/* Empty states */
.empty { text-align:center; padding:3.5rem 1rem; }
.empty h3 { font-size:1.5rem; letter-spacing:-.02em; font-weight:600; margin:1rem 0 .5rem; color:#f4f4f6; }
.empty p { color:var(--mute); max-width:30rem; margin:0 auto; line-height:1.6; }

@media (max-width: 768px) { .block-container { padding:1.2rem 1rem 4rem; } [class*="st-key-bezel"] { padding:1.2rem 1rem; } }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        '<div class="hero rise"><span class="eyebrow">EchoProof AI · Project Spectra</span>'
        "<h1>Listen closer.<br><span>See what the audio hides.</span></h1>"
        "<p>Local audio deepfake screening. Your recording is decoded and analyzed on this machine; "
        "scores are research-grade model outputs, not forensic proof.</p></div>",
        unsafe_allow_html=True,
    )


def card_title(text: str) -> None:
    st.markdown(f'<div class="card-h"><i></i>{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><span class="eyebrow">Not available yet</span><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True)
