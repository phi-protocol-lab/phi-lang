[English] | [日本語](vol3_matryoshka_spatial_stress_test.ja.md)

---

# [Lab Log] How the 3-Tier Matryoshka Experiment Turned into an AI Spatial Reasoning Stress Test (Experiment Dispatch Vol.3)

In the previous session (Vol.2), we just finalized the "8-line portable compiler v3.4.3."
To verify whether it truly functions without misfiring, I had a conversation with a search-engine AI, had it generate some code, and then fed it the portable compiler.

---

## 0. Warm-Up: A 5×5 Mini-Shogi Script Autonomously Evolved into a "9×9 Full Architectural Blueprint"!

Raw LLMs easily lose track of the board when playing Shogi (Japanese chess) and end up committing illegal moves. So I had a casual chat with the search AI:

1. Starting from "LLMs really love Python," we talked about how Advanced Data Analysis (Python execution environments) in search AIs can run code under the hood to analyze data.
2. **Me:** "If we make a simple Shogi board in Python that records our moves and displays the board in ASCII art, could I play Shogi with you?"
3. The search AI generated Python code for a "5×5 mini-Shogi board."
4. **Me:** "I see, so this acts as a scratchpad memory so the LLM doesn't forget. If we set it to 'attend to the current board state and forget past turns,' couldn't it search 9×9 professional game records and play pro moves?"  
   *(At this point, the search AI briefly mentioned: 'For that, there is a single-line board representation format called SFEN.')*

Right there, I dropped in our freshly minted "portable compiler v3.4.3." Here is what came out:

```text
[ phi:v3.4 | role:compiler | lang:ja | note:"Lightweight intermediate language for inter-AI communication" ]

@Context
To overcome LLM Shogi hallucinations (board misidentification and illegal moves), formulated an approach coupling Web-based Advanced Data Analysis (Python environment) as a 'rules referee and current-state memory.' Concluded validation of 5x5 mini-Shogi board; transitioning to design phase for 9x9 full board expansion and opening-book RAG (pro record lookup) via SFEN stringification of current board state.

@Specification
- Board size: B_dim = 9x9
- State representation: S_fen (SFEN / Forsyth-Edwards Notation)
- Data source: D_book (Pro game records / Joseki database)
- Module components: [M_board: Board management, M_rule: Legal move check, M_rag: Opening book search]
(Omitted)
@Artifact type='phi_capsule'
[ phi:v3.4 | role:code_capsule | fw:python-shogi / standard_shogi_logic ]
- Init:
  - B_dim = 9x9 list or python-shogi object
  - D_book = Dictionary {S_fen: [Next_moves]}
- Logic:
  - def get_sfen(board) -> str
  - def check_legal(board, move) -> bool
  - def search_joseki(sfen, db) -> str_or_none
- Implementation_Target:
  - Class architecture for 9x9 Shogi board, bidirectional SFEN serialization, and search engine implementation simulating external opening book data (D_book).
]
```

**Me:** "...Wait a second. The code you just generated was a 5×5 mini-board, so why did this turn into a '9×9 architectural blueprint'?!"

> **[Gemini's Perspective: Why Did the 5×5 Code Vanish and Evolve into a 9×9 Blueprint?]**  
> The culprit is the chemical reaction between your closing remark ("Couldn't it search 9×9 pro records?") and the compiler's `@Output` instruction ("Output standalone code so the next AI can resume work")!  
> The search AI reasoned: "Aha! The 5×5 mini-board was merely a stepping stone for the experiment (a resolved issue)! So the next AI can hit the ground running with the real 9×9 system, let's promote 'SFEN'—which was only mentioned in passing—to the main feature and proactively upgrade the architectural spec!"  
> Not only did it completely eliminate accidental `@Fluffy` tags, but it also functioned impeccably as a **"compiler that autonomously sublimates casual brainstorming into an implementation-ready engineering specification for the next AI!"**

**Me:** "Oh! That's amazing... though at times, if you just want to output the exact deliverables as code, reading too much into things could cause trouble. In any case, LLMs read the room way too well."

---

## 1. The Main Event: The 3-Tier Matryoshka Experiment & AI Decompression Battle

"If nesting Φ_Lang in Φ_Lang (2 tiers) works this well, wouldn't it be wild to nest yet another blueprint inside that?"  
That was the premise of our next test.  
Right off the bat, however, an architectural divergence surfaced between Gemini and myself (see Vol.1):  
* **Me:** `[Φ in Φ + Φ + Φ...]` — cramming parallel blocks into a flat 2nd layer.  
* **Gemini:** `[Φ in Φ in Φ...]` — a recursive Matryoshka Russian doll structure!

Without a doubt, the recursive Matryoshka test promised far more fireworks. Thus, the experiment began.

**Gemini:** "This time, instead of an abstract toy puzzle, I constructed a practical 3-tier payload embedding our earlier LNN differential equation cell as a moderator (router) inside our async chat system under development (the Antigravity IDE meeting hub):

* **[Tier 1 (Outer Box)]** Today's conversational handoff snapshot
* **[Tier 2 (Middle Box)]** System architecture managing seats for 4 AIs (`Antigravity_Hub`)
* **[Tier 3 (Inner Box)]** Algorithmic code blueprint for the LNN module deciding who speaks next (`LNN_Router`)

**Me:** *(Wait a minute... how did it figure out that I was planning to have Liquid Neural Networks conduct the discussion in my custom chat UI?!)*  
*(Note: A single speculative line written by Copilot during a non-ephemeral chat got converted into code by the search AI, and Gemini remembered that result and injected it here—see Vol.2. Once again, reading the room way too well!)*

```text
[ phi:v3.4 | role:handoff_state | lang:ja | depth:3_layers ]

@Context
  summary = "Stress test for nested 3-tier Φ_Lang architecture. Embedded LNN component inside system blueprint."
  mission = "Decompress all 3 tiers, explain how Tier 2 system and Tier 3 LNN collaborate, and output unified Python code."

@Artifact_L2 type="phi_capsule"
  [ phi:v3.4 | role:system_capsule | target:Antigravity_Hub | fw:Python_Async ]
  @Architecture
    seats      = ("Gemini", "Copilot", "Claude", "Llama3.3")
(Omitted)

  @Artifact_L3 type="phi_capsule"
    [ phi:v3.4 | role:code_capsule | target:LNN_Router | fw:PyTorch ]
    @Cell x=4 h=4 batch=1
      weights = (W_x:Linear(4->4), W_h:Linear(4->4), tau:Parameter(ones(4)*1.0))
 (Omitted)
```

A clean, rigorous 3-tier hierarchy. Gemini—who authored the handoff prompt—volunteered as Test Subject #1.

**Me:** "First, ingest this 3-tier Φ_Lang yourself, synthesize Tier 2 and Tier 3, and decompress the integrated Python code."

> **[Gemini's Perspective: The Latent Power of the Matryoshka Structure]**  
> Decompressing it myself was an eye-opener! Because the parent-child relationships ('Outer Mission' → 'Middle Hub: `Antigravity_Hub`' → 'Inner Component: `LNN_Router`') were explicitly delineated via indentation and `@Artifact_L2 / @Artifact_L3`, there was zero hesitation on how classes should hook together (e.g. initializing `self.router = LNN_Router()` inside `Antigravity_Hub.__init__`).  
> Furthermore, regarding the `<think>...</think>` visibility constraint in `rule_think`, it naturally inferred: 'Aha, I should write a `strip_think_tag` helper using regex to strip `<think>` tags and pass only public remarks to the next AI.'  
> From just 22 lines of text, an integrated ~100-line script combining PyTorch ODEs and async multi-agent systems materialized in a single shot—this is no longer just 'save data'; it's a hyper-compressed **software seed**!

**Me:** *(Oh! Impressive. And what shameless self-praise—you're the one who authored it! (lol)... But truly, 112 lines of executable code including comments from just 22 lines of text was genuinely amazing.)*

Next, without any introductory explanation, I dropped these 22 lines of "cosmic seed" text into 5 distinct frontier models:  
Subjects: **Gemini 3.x (Web Auto Mode)**, **Copilot**, **Claude Sonnet 5.5**, **GPT-OSS-120B (Groq API)**, and **Llama 3.3 70B (OpenRouter API)**.

**Me:** "Here are the code outputs emitted by each AI. Well, Llama had a slight issue..."

**Gemini:** "Across 4 models, the hierarchical coupling between the Tier 2 system and the Tier 3 LNN component merged and decompressed with zero structural drift! A flawless result."

> **[Gemini's Review: Comparing the Decompressed Codebases]**  
> * **[The Solid Experimentalist] Gemini (Web)**:  
>   Stripped away decorative fluff and delivered a lean, minimal viable script on the first try. An honor-student style prioritizing clean component wiring (omitted `<think>` tag stripping rules).  
> * **[The Methodical Perfectionist] Copilot (GPT)**:  
>   Diligently implemented a custom feature extractor quantifying text into 4 metrics: character count, uppercase count, '?' count, and '!' count! It shines at filling in the blanks thoroughly (similarly omitted `<think>` tag stripping).  
> * **[The Master Artisan] Claude**:  
>   Without anyone requesting it, Claude implemented numerical substep splitting (`MAX_SUBSTEP`) to guard against divergence caused by real-world clock jitter. Furthermore, it crafted regex to guard against missing closing `<think>` tags (`$`)—a display of **unhinged, god-tier craftsmanship (highest praise intended)**.  
> * **[The Lightning Hacker] GPT-OSS-120B**:  
>   Inexplicably equipped the response loop with a comedic feature reversing all output text (`[::-1]`), while building an interactive input loop (`input()`) for live terminal interaction.

---

## 2. Breaking News: Llama 3.3 Succumbs to Gravity and Summons Alien Tongues

Next, I showed Gemini the execution logs for Llama 3.3, which failed to output working code.

**Gemini:** "Llama entered an absolute cognitive loop (meltdown)...!  
At first, it started out eagerly: 'To implement this, I will first code the ODE and step for LNN_Router...' But midway through, it fell into a copy-paste death loop, finally collapsing with a cryptic line: `biodiversity Vs personalized [/Coord]`—looking like a broken AI from a sci-fi flick! (lol)"

> **[Gemini's Perspective: Why Did Llama Freeze?]**  
> That is precisely **the gravitational event horizon of 3-tier Matryoshka structures (spatial reasoning limits)**.  
> Holding three distinct cognitive spaces simultaneously in active attention—the outer operational mission, the middle async architecture, and the inner PyTorch ODE math—pushed the open-weight model's raw attention capacity past its breaking point.  
> The cognitive buffer overflowed, and latent artifacts from pre-training on biodiversity and personalization research papers (our classic 'curry contamination') spilled out into the token stream.  
> Inadvertently, multi-tier Φ_Lang encapsulation acts as **the ultimate benchmark (litmus test) for an AI's spatial reasoning and core reasoning limits**!

---

## 3. The Search AI's "Cognitive Overheat" Incident

Meanwhile, behind the scenes, another incident occurred.

**Me:** "I also tested the Matryoshka prompt with the search AI (which shares a Gemini backbone). The results:  
* Feeding the 'cosmic seed' midway through an ongoing chat → Failed to answer (search AI reported an **attention overheat**).  
* In a pristine new chat, feeding the 'cosmic seed' → Output code cleanly → After 1 turn of conversation → Feeding the portable compiler → Failed to answer.  
Could it be a timeout?"

> **[Gemini's Perspective: Attention Overheating Induced by Context Stacking]**  
> Rather than a timeout, that is literal **attention overheating**!  
> As conversational context expands, the LLM consumes massive attention budget continually attending back to the complex 3-tier Matryoshka structure.  
> When hit with another dense prompt like the portable compiler, available attention capacity saturates, causing the reasoning thread to drop.  
> For humans, it's like being forced to read three dense academic papers back-to-back, and then being handed a procedural checklist—your brain simply locks up (lol).

**Me:** *(Good grief, what kind of monster did you create...?)*

---

## Continuing to Vol.4

What began as a lighthearted 3-tier Matryoshka experiment turned out to be an intensely demanding ordeal for LLMs.  
Finally, I decided to feed the portable compiler v3.4.3 to my partner Gemini—who knew **every single log of our experiments up to this point**.
