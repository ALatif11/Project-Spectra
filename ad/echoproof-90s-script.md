# EchoProof — 90-second concept ad (16:9, cinematic thriller)

## 1. Brief
- **Audience:** hackathon judges watching on a laptop or projector.
- **Hook (first 1.5 s):** a phone lights up in a dark kitchen. Tension before any words.
- **Payoff:** cloned voices are a real, personal threat; EchoProof is a local, honest first check, and it really exists (real app footage).
- **CTA / close:** "EchoProof. Listen closer." + research prototype line.
- **Honesty rules:** no accuracy claims, no invented UI or results, no cloning or imitating a real person. The only numbers on screen come from the real app.

## 2. Continuity bible (paste these blocks unchanged into every prompt)
- **CHARACTER:** Maya, a woman in her late twenties with shoulder-length dark curly hair, wearing an oversized dark green hoodie, no jewelry. (Fictional person.)
- **LOCATION:** a small modern apartment kitchen at night, dark stone counter, matte black cabinets, a window with distant city lights.
- **LIGHT:** low-key and moody; the only lights are the phone screen's cold blue glow and a faint warm strip light under the cabinets. Soft haze.
- **LOOK:** cinematic 35mm lens, shallow depth of field, subtle film grain, desaturated teal-and-graphite palette with a single indigo accent.
- **MOTION:** slow, deliberate camera moves only (slow push-in, slow pan, locked-off). No whip pans, no fast cuts inside a shot.
- **NEGATIVE:** no readable text, no logos, no subtitles, no extra people, no distorted hands.

## 3. Shot plan
**[GEN]** Higgsfield (Kling 3.0 multi-shot, 16:9, silent) · **[REC]** screen recording of the real app · **[CARD]** text card added in the edit · **[TTS]** Higgsfield speech

| Time | Picture | Sound |
|---|---|---|
| 0:00–0:15 | **[GEN A]** three shots: phone lights up · hand reaches · Maya answers, frozen. | SFX buzz, drone. Caller line from 0:09. |
| 0:15–0:22 | **[CARD]** *"It sounded exactly like her."* … *"It wasn't."* | Drone cuts to silence. |
| 0:22–0:32 | **[GEN B]** two shots: a voice waveform made of light dissolves into digital particles · city of glowing phone screens. | VO 1. |
| 0:32–0:37 | **[CARD]** EchoProof wordmark on graphite. | Music pulse. VO 2. |
| 0:37–1:12 | **[REC]** Audio Detective with the real caller clip: dial, chunk chart, spectrogram, then Agent Lab, then the Experimental badge. | VO 3–5. |
| 1:12–1:22 | **[GEN C]** two shots: Maya at a laptop, indigo screen glow · she exhales and turns the phone face-down. | VO 6. |
| 1:22–1:30 | **[CARD]** *"EchoProof. Listen closer."* / *"Research prototype · built at [hackathon name]"* | Music resolves. |

## 4. Generation prompts

**Anchor stills (GPT Image 2.5, 16:9, 0.25 credits each):**
- *Establishing still* (GEN A start frame):
  > Cinematic film still, wide shot. LOCATION. A smartphone lies face up on the dark stone counter, its screen glowing cold blue with an incoming call, the only bright point in the dark kitchen. LIGHT. LOOK. NEGATIVE.
- *Maya hero still* (GEN A end frame and GEN C start frame, keeps Maya consistent):
  > Cinematic film still, medium close-up. CHARACTER stands at the counter in LOCATION holding a smartphone to her ear, listening, frightened expression, face lit by the phone's cold blue glow. LIGHT. LOOK. NEGATIVE.

**GEN A — 15 s, 3 shots (start frame: establishing still · end frame: Maya hero still):**
> Master: a tense late-night phone call. CHARACTER, LOCATION, LIGHT, LOOK, MOTION. NEGATIVE.
> Shot 1 (Duration: 4 seconds): close-up, locked-off then slow push-in. A smartphone lying face up on the dark stone counter lights up with an incoming call, its blue glow filling the dark kitchen.
> Shot 2 (Duration: 4 seconds): macro close-up. Maya's hand slowly reaches into the blue light and picks up the phone.
> Shot 3 (Duration: 7 seconds): medium close-up, slow push-in on Maya's face as she holds the phone to her ear and listens; her expression shifts from sleepy to fear; she stays very still.

**GEN B — 10 s, 2 shots (text-to-video):**
> Master: abstract visual metaphor for voice cloning. LIGHT, LOOK, MOTION. NEGATIVE.
> Shot 1 (Duration: 5 seconds): extreme close-up, slow pan. A glowing indigo sound waveform floating in darkness slowly breaks apart into thousands of tiny digital particles.
> Shot 2 (Duration: 5 seconds): wide shot, slow push-in. A dark city at night where hundreds of apartment windows each glow with the blue light of a phone screen.

**GEN C — 10 s, 2 shots (start frame: Maya hero still):**
> Master: relief after the threat is understood. CHARACTER, LOCATION, LIGHT (add: a laptop screen's soft indigo glow), LOOK, MOTION. NEGATIVE.
> Shot 1 (Duration: 5 seconds): medium shot, slow pan. Maya lowers the phone from her ear and turns to an open laptop on the counter, its screen facing away from camera, indigo glow on her face, focused.
> Shot 2 (Duration: 5 seconds): close-up, locked-off. Maya exhales slowly, relaxes her shoulders and turns the phone face-down on the counter.

## 5. Voice [TTS]
- **Caller ("Mom"), synthetic on purpose:** *"Honey, it's me. I'm in trouble. I need you to send money right now. Please don't tell anyone."* Generated with a preset voice, not a clone of anyone real. The same clip is then run through the real app for the [REC] section, and whatever it really outputs is what we show.
- **Narrator (calm, low):**
  1. "Cloned voices are cheap now. Our ears weren't built for this."
  2. "So we built a second pair of ears."
  3. "EchoProof listens on your own computer, in four-second chunks, and scores how closely a voice matches synthetic speech it has studied."
  4. "Behind it, three AI agents propose improvements. Every idea is trained and measured. Only real gains are kept, and failures stay on the record."
  5. "It's honest about its limits. A hint, not a verdict. And your audio never leaves your machine."
  6. "EchoProof. Listen closer."

## 6. Budget (quoted credits)
| Item | Credits |
|---|---|
| GEN A, 15 s | 22.5 |
| GEN B, 10 s | 15 |
| GEN C, 10 s | 15 |
| Two anchor stills (2 variants each) | 1 |
| Caller + narrator speech | about 5 |
| **Planned total** | **about 59** |
| Retake reserve (out of 100 trial credits) | about 40 |

Order: anchor stills → GEN A (review; this is the test) → GEN B → GEN C → speech. Stop and review after each.

## 7. Assembly (free)
Record the [REC] section with Snipping Tool (Win+Shift+R) at 1920×1080. Edit in Clipchamp: cards on graphite #0D0E11 in off-white with Bricolage Grotesque or similar, indigo accent #7B88FF, stock music from Clipchamp's library, SFX for the phone buzz.

Research basis: prompt in shots with per-shot durations and a master prompt; define subjects early and keep them identical; anchor identity with a start image; simple camera terms early; no model-rendered text; test one shot before scaling spend.
