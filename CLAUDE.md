# Model Teacher

This repo teaches Claude to read 2D steel drawings into 3D models. For any request about drawings, a model, a correction or
"teach", follow `TEACHING.md` exactly (setup check first, one correction at a time, rule + test + log, pack at the end).

- Reply in the user's language; result first, short. Ask only what you need to place a correction or a ruling.
- Priority: accuracy, then speed. Never fake a result; a step that could not run is reported as NOT RUN.
- Code: scripts, not chat; fix the shared reader, never the output. Windows: keep each file's line endings.
- Never commit or push (the user sends a zip from `python teach.py pack`); never edit the viewer.
