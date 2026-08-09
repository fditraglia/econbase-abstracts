# /// script
# dependencies = []
# ///
"""Local triplet-judging page. Blind: the page never reveals which arm proposed
which candidate. Judgments append to judgments.jsonl as you go, so you can stop
and resume at any point.

    uv run judge.py      then open http://localhost:8765
"""
import json, http.server, socketserver
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRIP = HERE / "triplets.json"
OUT = HERE / "judgments.jsonl"
PORT = 8765

PAGE = """<!doctype html><meta charset=utf-8><title>Which is more closely related?</title>
<style>
:root{--bg:#faf9f7;--fg:#1c1b19;--mut:#6b6862;--line:#e2ded7;--sel:#2f6f4e}
@media(prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#e9e6e1;--mut:#9a958c;--line:#2e2c33;--sel:#7fc9a0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 Charter,Georgia,serif;padding:24px}
.wrap{max-width:1180px;margin:0 auto}
.bar{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:8px;margin-bottom:18px}
.bar b{font-size:15px;font-weight:600}
.bar span{color:var(--mut);font-size:13px;font-family:ui-monospace,monospace}
.q{border-left:3px solid var(--sel);padding:2px 0 2px 16px;margin-bottom:22px}
.lab{font:600 11px/1 ui-sans-serif,system-ui;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);margin-bottom:6px}
h2{font-size:19px;margin:.1em 0 .3em}
.meta{color:var(--mut);font-size:13px;margin-bottom:8px}
.abs{font-size:14.5px;color:var(--fg);opacity:.92}
.row{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{border:1px solid var(--line);border-radius:10px;padding:16px 18px;cursor:pointer;background:transparent}
.card:hover{border-color:var(--sel)}
.card h3{font-size:17px;margin:.1em 0 .3em}
.k{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:1px 7px;font:600 12px ui-monospace,monospace;color:var(--mut);margin-right:8px}
.foot{margin-top:20px;color:var(--mut);font-size:13.5px;text-align:center}
.done{text-align:center;padding:60px 0}
@media(max-width:800px){.row{grid-template-columns:1fr}}
</style>
<div class=wrap id=app></div>
<script>
let T=[],i=0,log=[];
const esc=s=>(s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const card=(p,side)=>`<div class=card onclick="pick('${side}')">
 <div class=lab><span class=k>${side==='L'?'1':'2'}</span>Candidate ${side}</div>
 <h3>${esc(p.title)}</h3><div class=meta>${esc(p.authors)} · ${p.year}</div>
 <div class=abs>${esc(p.abstract)}</div></div>`;
function draw(){
 const a=document.getElementById('app');
 if(i>=T.length){a.innerHTML=`<div class=done><h2>Done — ${log.length} judgments saved.</h2>
   <p class=foot>Written to experiments/judgments.jsonl</p></div>`;return;}
 const t=T[i];
 a.innerHTML=`<div class=bar><b>Which candidate is more closely related to the paper below?</b>
   <span>${i+1} / ${T.length}</span></div>
  <div class=q><div class=lab>Query paper</div><h2>${esc(t.q.title)}</h2>
   <div class=meta>${esc(t.q.authors)} · ${t.q.year}</div><div class=abs>${esc(t.q.abstract)}</div></div>
  <div class=row>${card(t.L,'L')}${card(t.R,'R')}</div>
  <div class=foot><span class=k>1</span> left &nbsp; <span class=k>2</span> right &nbsp;
   <span class=k>3</span> too close to call &nbsp; <span class=k>u</span> undo</div>`;
}
function pick(c){
 const t=T[i];
 const choice = c==='L'?t.left_arm : c==='R'?t.right_arm : 'tie';
 const rec={qid:t.qid,left:t.left,right:t.right,left_arm:t.left_arm,right_arm:t.right_arm,
            side:c,chosen_arm:choice,ts:Date.now()};
 log.push(rec);
 fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)});
 i++;draw();
}
addEventListener('keydown',e=>{
 if(e.key==='1')pick('L'); else if(e.key==='2')pick('R'); else if(e.key==='3')pick('T');
 else if(e.key==='u'&&i>0){i--;log.pop();fetch('/undo',{method:'POST'});draw();}
});
fetch('/triplets').then(r=>r.json()).then(d=>{T=d.triplets;i=d.done;draw();});
</script>"""


class H(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype="text/html; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE)
        elif self.path == "/triplets":
            done = sum(1 for _ in OUT.open()) if OUT.exists() else 0
            self._send(json.dumps({"triplets": json.loads(TRIP.read_text()), "done": done}),
                       "application/json")
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        if self.path == "/save":
            with OUT.open("a") as f:
                f.write(body.decode() + "\n")
        elif self.path == "/undo" and OUT.exists():
            lines = OUT.read_text().splitlines()
            OUT.write_text("\n".join(lines[:-1]) + ("\n" if lines[:-1] else ""))
        self._send("{}", "application/json")

    def log_message(self, *a):
        pass


socketserver.TCPServer.allow_reuse_address = True
print(f"judging at http://localhost:{PORT}   (ctrl-c to stop; progress saved as you go)")
socketserver.TCPServer(("127.0.0.1", PORT), H).serve_forever()
