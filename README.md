# Model Teacher

Teach Claude to turn 2D steel structure drawings (PDF) into a 3D model. You give it your drawings, it builds the model,
you correct what it read wrong. Every correction becomes a general reading rule with a test, so the next drawing is read
right. When you are happy with the model, Claude packs what you taught into one zip; send it back to the person who shared
this link. The rules from every user are compared and the best ones are merged into the next version of this kit.

## Start with Claude Code (recommended)
1. Install Python 3.10+ and [Claude Code](https://claude.com/claude-code).
2. `git clone https://github.com/D4rKLoV3-09/Model-Teacher.git` (or Code > Download ZIP and unzip), open the folder in
   Claude Code.
3. Say: **"Teach: my drawings are `<path to the PDF>`"**. Claude installs what it needs (`python teach.py setup`) and starts.

## Start in a claude.ai chat
1. Settings > Capabilities: turn on **Code execution and file creation**, and allow **network egress** to package managers
   (needed once per chat to install PyMuPDF). Changing it only affects new chats.
2. On this page: Code > **Download ZIP**.
3. New chat: attach the ZIP and your drawing PDF, and paste:
   > Unzip Model-Teacher and follow TEACHING.md in it. My drawings are attached.
4. At the end, download the zip Claude gives you. A chat cannot keep files between chats: finish one building in one chat,
   or keep the zip and attach it again to continue.

## What you will see
- **The model page** (an HTML file): rotate, select members, read sizes and marks; opens in any browser.
- **The ruling sheet**: every reading Claude was unsure of, grouped by kind, with a crop of your drawing beside each option.
  Answer a question by its number ("5 B") or say in words what is wrong ("this line is a dimension, not a brace").
- After each correction: what changed (members, tonnes) and what the model still misses.

## Your drawings
Your drawings stay on your PC / in your chat; the kit uploads nothing. The zip you send back holds your corrections, the
changed reading code, the job settings and small crops of your drawings that show each taught case (the tests). Check it
before sending if your drawings are confidential.

## Layout
`teach.py` (setup / start / build / note / answer / status / pack), `TEACHING.md` (the loop Claude follows),
`generate-model/` (readers, model builder, viewer), `rules/` (rulings and drawing styles), `jobs/` (your work, not in git).
